import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import logging
from .policy_context import PolicyContext
from .rm_agent import RMAgent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LeaveAgent")


class LeaveApplicationState(TypedDict):
    leaveData: Dict[str, Any]
    applicationId: Optional[str]
    status: str
    conflicts: List[str]
    approvalStatus: str
    finalMessage: str
    error: Optional[str]


class LeaveAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        # In-memory storage for leave applications (in production, use a database)
        self.leave_applications: Dict[str, Dict[str, Any]] = {}
        
        # Initialize policy context
        self.policy_context = PolicyContext(os.path.join(os.path.dirname(__file__), "..", "docs"))
        
        # Initialize RM Agent for supervisor space lookup
        self.rm_agent = RMAgent()
        
        # Build the LangGraph workflow
        self.workflow = self.build_workflow()
        
        # Optional Google API config (tools)
        self.service_account_file = os.getenv("SERVICE_ACCOUNT_KEY_FILE", "./service-account-key.json")
        self.calendar_scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
        self.chat_scopes = [
            "https://www.googleapis.com/auth/chat.bot",
            "https://www.googleapis.com/auth/chat.messages",
            "https://www.googleapis.com/auth/chat.spaces",
        ]
        logger.info("LeaveAgent initialized.")

    async def _llm_text(self, system: str, payload: dict) -> str:
        try:
            from datetime import datetime, timezone
            today_iso = datetime.now(timezone.utc).date().isoformat()
            system_with_date = (
                f"{system}\n\n"
                f"Context: Today's date is {today_iso}. Interpret any relative dates (e.g., 'next Monday') relative to today."
            )
            messages = [
                SystemMessage(content=system_with_date),
                HumanMessage(content=json.dumps(payload)),
            ]
            resp = await self.llm.ainvoke(messages)
            return (resp.content or "").strip()
        except Exception:
            return ""
    
    def get_state_definition(self):
        """Define the state structure for the workflow"""
        return {
            "leaveData": lambda x, y: y if y is not None else x,
            "applicationId": lambda x, y: y if y is not None else x,
            "status": lambda x, y: y if y is not None else x,
            "conflicts": lambda x, y: y if y is not None else x,
            "approvalStatus": lambda x, y: y if y is not None else x,
            "finalMessage": lambda x, y: y if y is not None else x,
            "error": lambda x, y: y if y is not None else x,
        }
    
    async def validate_and_create_application(self, state: LeaveApplicationState) -> LeaveApplicationState:
        """Step 1: Validate and create leave application"""
        try:
            leave_data = state["leaveData"]
            
            # Generate unique application ID
            application_id = str(uuid.uuid4())
            
            # Validate required fields
            required_fields = ["startDate", "endDate", "leaveType", "reason"]
            missing_fields = [
                field for field in required_fields
                if not leave_data.get(field) or str(leave_data[field]).strip() == ""
            ]
            
            if missing_fields:
                # Special handling for supervisor email
                if "supervisorEmail" in missing_fields:
                    prompt = "I need your supervisor's email address to send the approval request. Please provide it (e.g., john@company.com)."
                else:
                    prompt = await self._llm_text(
                        "You are a friendly HR assistant. Ask for the missing leave fields in ONE short sentence (<=25 words).",
                        {"missing": missing_fields, "current": leave_data},
                    )
                    if not prompt:
                        prompt = f"Please provide: {', '.join(missing_fields)}."
                return {**state, "error": "missing_fields", "finalMessage": prompt, "status": "failed"}
            
            # Validate supervisor email format
            supervisor_email = leave_data.get("supervisorEmail", "")
            if supervisor_email and "@" not in supervisor_email:
                return {**state, "error": "invalid_email", "finalMessage": "Please provide a valid supervisor email address (e.g., john@company.com).", "status": "failed"}
            
            # Validate against company policy using LLM
            policy_validation = await self.policy_context.validate_leave_against_policy(leave_data, self.llm)
            if not policy_validation["valid"]:
                conflicts = policy_validation.get("conflicts", [])
                warnings = policy_validation.get("warnings", [])
                
                conflict_message = "I found some policy conflicts with your leave request:\n" + "\n".join(f"• {conflict}" for conflict in conflicts)
                if warnings:
                    conflict_message += "\n\nAdditional considerations:\n" + "\n".join(f"• {warning}" for warning in warnings)
                
                return {**state, "error": "policy_conflict", "finalMessage": conflict_message, "status": "failed"}
            
            # Add policy warnings to the application if any
            policy_warnings = policy_validation.get("warnings", [])
            
            # Create leave application record
            application = {
                "id": application_id,
                **leave_data,
                "submittedAt": datetime.now().isoformat(),
                "status": "pending",
                "requesterEmail": leave_data.get("requesterEmail"),
                "supervisorEmail": supervisor_email,
            }
            
            self.leave_applications[application_id] = application
            
            logger.info(f"Creating leave application for: {leave_data.get('requesterEmail')}")
            
            return {
                **state,
                "applicationId": application_id,
                "status": "created",
            }
        except Exception as error:
            logger.error(f"Error in validate_and_create_application: {error}", exc_info=True)
            apology = await self._llm_text(
                "Write a brief, polite apology and ask the user to try again or share details.",
                {"stage": "create_application", "error": str(error)},
            )
            return {**state, "error": str(error), "finalMessage": apology or "Sorry, something went wrong. Could you try again?", "status": "failed"}
    
    async def check_calendar_conflicts(self, state: LeaveApplicationState) -> LeaveApplicationState:
        """Step 2: Check for calendar conflicts (always use Google Calendar tool first)"""
        try:
            leave_data = state["leaveData"]

            # Always use Google Calendar tool
            tool_conflicts = await self.try_check_google_calendar(leave_data)
            logger.info(f"Google Calendar conflicts: {tool_conflicts}")
            if tool_conflicts:
                return {
                    **state,
                    "conflicts": tool_conflicts,
                    "status": "conflicts_found",
                }

            # If no conflicts found, fallback to LLM reasoning
            analysis = await self._llm_text(
                "You are a scheduling assistant. Based on leave details, list potential conflicts (like long duration, month-end closings). Return newline-separated bullets; or 'none'.",
                leave_data,
            )
            conflicts: List[str] = []
            if analysis and analysis.lower().strip() != "none":
                for line in analysis.splitlines():
                    line = line.strip("- • \t ")
                    if line:
                        conflicts.append(line)
            if not conflicts:
                conflicts = await self.simulate_calendar_check(leave_data)

            logger.info(f"Checking calendar conflicts for application: {state.get('applicationId')}")

            return {
                **state,
                "conflicts": conflicts,
                "status": "conflicts_found" if conflicts else "no_conflicts",
            }
        except Exception as error:
            logger.error(f"Error in check_calendar_conflicts: {error}", exc_info=True)
            apology = await self._llm_text(
                "Briefly explain you couldn't analyze conflicts and ask user to proceed or adjust dates.",
                {"stage": "conflicts", "error": str(error)},
            )
            return {**state, "error": str(error), "finalMessage": apology or "Couldn't analyze conflicts. Do you want to proceed?", "status": "failed"}

    async def process_approval_workflow(self, state: LeaveApplicationState) -> LeaveApplicationState:
        """Step 3: Process approval workflow (always send supervisor poll)"""
        try:
            leave_data = state["leaveData"]
            conflicts = state["conflicts"]
            application_id = state["applicationId"]

            logger.info(f"Processing approval workflow for application: {state.get('applicationId')}")

            if conflicts:
                # Format conflicts as plain sentences (no bullets/markdown)
                formatted_conflicts = []
                for conflict in conflicts:
                    text = conflict.strip().lstrip("-• ")
                    if not text.endswith("."):
                        text = text + "."
                    formatted_conflicts.append(text)

                message = await self._llm_text(
                    "Compose a concise message in plain text (no markdown): request submitted, calendar conflicts require manual review. List the conflicts clearly as sentences. End with polite expectation setting.",
                    {"applicationId": application_id, "conflicts": formatted_conflicts, "leave": leave_data},
                )
                if not message:
                    message = "Your request was submitted, but some calendar conflicts need manual review. We'll update you soon. " + " ".join(formatted_conflicts)
                # Always send supervisor poll even if conflicts exist
                poll_status = await self.try_send_supervisor_poll(leave_data, application_id)
                message += f"\n\nSupervisor poll sent: {poll_status}"
                return {**state, "approvalStatus": "requires_manual_review", "finalMessage": message, "status": "completed"}

            # Always send supervisor poll card (tool), if supervisor email present
            poll_status = await self.try_send_supervisor_poll(leave_data, application_id)

            # LLM-crafted approval message (status kept as pending_approval)
            try:
                start = leave_data.get("startDate")
                end = leave_data.get("endDate")
                leave_type = leave_data.get("leaveType")
                reason = leave_data.get("reason")
                payload = {"start": start, "end": end, "type": leave_type, "reason": reason}
                msg = await self._llm_text(
                    "Confirm leave submission in a friendly tone and state it's pending supervisor approval. Include start, end, type, and a short duration if obvious. If a supervisor poll was sent, mention it.",
                    {**payload, "poll": poll_status},
                )
                approval_result = {"status": "pending_approval", "message": msg or "Your request was submitted and is pending supervisor approval."}
            except Exception:
                approval_result = await self.simulate_approval_process(leave_data)

            # Update application status
            if application_id and application_id in self.leave_applications:
                application = self.leave_applications[application_id]
                application["approvalStatus"] = approval_result["status"]
                application["approvedAt"] = datetime.now().isoformat()
                self.leave_applications[application_id] = application

            enhanced_message = f"{approval_result['message']}\n\nReference ID: {application_id}"
            return {
                **state,
                "approvalStatus": approval_result["status"],
                "finalMessage": enhanced_message,
                "status": "completed",
            }
        except Exception as error:
            logger.error(f"Error in process_approval_workflow: {error}", exc_info=True)
            apology = await self._llm_text(
                "Briefly apologize and ask the user to retry the approval step later.",
                {"stage": "approval", "error": str(error)},
            )
            return {**state, "error": str(error), "finalMessage": apology or "Something went wrong during approval. Please try again.", "status": "failed"}
    
    async def simulate_calendar_check(self, leave_data: Dict[str, Any]) -> List[str]:
        """Simulate calendar conflict checking"""
        conflicts = []
        
        try:
            # Simulate conflicts for vacation leave
            if "vacation" in leave_data.get("leaveType", "").lower():
                start_date = datetime.strptime(leave_data["startDate"], "%Y-%m-%d") if leave_data["startDate"].count("-") == 2 else datetime.now()
                end_date = datetime.strptime(leave_data["endDate"], "%Y-%m-%d") if leave_data["endDate"].count("-") == 2 else datetime.now()
                duration = (end_date - start_date).days
                
                # Simulate conflict for long vacations
                if duration > 7:
                    conflict_date = start_date + timedelta(days=3)
                    conflicts.append(f"Long vacation period overlaps with team meeting on {conflict_date.strftime('%Y-%m-%d')}")
            
            # Simulate conflicts for sick leave
            if "sick" in leave_data.get("leaveType", "").lower():
                # Simulate conflict during month-end
                start_date = datetime.strptime(leave_data["startDate"], "%Y-%m-%d") if leave_data["startDate"].count("-") == 2 else datetime.now()
                day = start_date.day
                
                if 25 <= day <= 31:
                    conflicts.append("Sick leave during month-end reporting period")
        
            logger.info(f"Simulating calendar check for leave type: {leave_data.get('leaveType')}")
        
        except Exception as error:
            logger.error(f"Error in simulate_calendar_check: {error}", exc_info=True)
        
        return conflicts
    
    async def simulate_approval_process(self, leave_data: Dict[str, Any]) -> Dict[str, str]:
        """Require manual supervisor approval for all leave types."""
        try:
            start_date = datetime.strptime(leave_data["startDate"], "%Y-%m-%d") if leave_data["startDate"].count("-") == 2 else datetime.now()
            end_date = datetime.strptime(leave_data["endDate"], "%Y-%m-%d") if leave_data["endDate"].count("-") == 2 else datetime.now()
            duration = max(1, (end_date - start_date).days + 1)
            logger.info(f"Simulating approval process for: {leave_data.get('requesterEmail')}")
            return {
                "status": "pending_approval",
                "message": (
                    f"Leave request submitted. "
                    f"Details: Start {leave_data['startDate']}, End {leave_data['endDate']}, "
                    f"Type {leave_data['leaveType']}, Duration {duration} day(s), Reason {leave_data['reason']}. "
                    "Your request is pending supervisor approval. You'll be notified once it's reviewed."
                ),
            }
        except Exception as error:
            logger.error(f"Error in simulate_approval_process: {error}", exc_info=True)
            return {
                "status": "error",
                "message": "Something went wrong while simulating the approval.",
            }
    
    def build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(LeaveApplicationState)
        
        # Add nodes
        workflow.add_node("validate", self.validate_and_create_application)
        workflow.add_node("check_conflicts", self.check_calendar_conflicts)
        workflow.add_node("process_approval", self.process_approval_workflow)
        
        # Add edges
        workflow.add_edge("validate", "check_conflicts")
        workflow.add_edge("check_conflicts", "process_approval")
        workflow.add_edge("process_approval", END)
        
        # Set entry point
        workflow.set_entry_point("validate")
        
        return workflow.compile()

    async def try_check_google_calendar(self, leave_data: Dict[str, Any]) -> List[str]:
        """Optional tool: Check user's Google Calendar for conflicts. Returns list of conflict descriptions or []."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            if not os.path.exists(self.service_account_file):
                return []
            creds = service_account.Credentials.from_service_account_file(self.service_account_file, scopes=self.calendar_scopes)
            service = build('calendar', 'v3', credentials=creds)

            user_email = leave_data.get("requesterEmail")
            start = leave_data.get("startDate")
            end = leave_data.get("endDate")
            if not (user_email and start and end):
                return []

            time_min = f"{start}T00:00:00Z"
            time_max = f"{end}T23:59:59Z"

            events_result = service.events().list(calendarId=user_email, timeMin=time_min, timeMax=time_max, singleEvents=True, orderBy='startTime').execute()
            items = events_result.get('items', [])
            conflicts: List[str] = []
            
            for ev in items:
                title = ev.get('summary') or 'Busy'
                start_time = (ev.get('start', {}) or {}).get('dateTime') or ev.get('start', {}).get('date')
                
                # Parse and format the time more descriptively
                try:
                    from datetime import datetime
                    if 'T' in start_time:
                        # DateTime format
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        formatted_time = dt.strftime("%I:%M %p on %A")
                        day_name = dt.strftime("%A")
                    else:
                        # Date only format
                        dt = datetime.fromisoformat(start_time)
                        formatted_time = dt.strftime("%A")
                        day_name = dt.strftime("%A")
                    
                    # Create more descriptive conflict message
                    conflicts.append(f"You have '{title}' at {formatted_time}")
                    
                except Exception as e:
                    # Fallback to original format if parsing fails
                    conflicts.append(f"Overlaps with: {title} on {start_time}")
            
            logger.info(f"Checking Google Calendar for: {leave_data.get('requesterEmail')}")
            
            return conflicts
        except Exception as error:
            logger.error(f"Calendar tool unavailable or failed: {error}", exc_info=True)
            return []

    async def try_send_supervisor_poll(self, leave_data: Dict[str, Any], application_id: Optional[str]) -> str:
        """Optional tool: Send a Google Chat card with Approve/Deny buttons to supervisor.
        Buttons link to APPROVAL_WEBHOOK_URL with applicationId and decision query params.
        
        Note: DM spaces don't contain user email information, so we use fallback to regular spaces.
        In production, maintain a database mapping supervisor emails to their DM space names.
        """
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            
            if not os.path.exists(self.service_account_file):
                logger.warning("Service account file not found")
                return "no_service_account"
                
            creds = service_account.Credentials.from_service_account_file(self.service_account_file, scopes=self.chat_scopes)
            chat_service = build('chat', 'v1', credentials=creds)

            supervisor_email = leave_data.get("supervisorEmail")
            if not supervisor_email or "@" not in supervisor_email:
                logger.warning(f"Invalid supervisor email: {supervisor_email}")
                return "invalid_supervisor_email"

            # Get supervisor's chatspace from database using RM Agent
            requester_email = leave_data.get("requesterEmail")
            supervisor_chatspace = None
            
            if requester_email:
                supervisor_chatspace = await self.rm_agent.get_supervisor_chatspace(requester_email)
            
            if supervisor_chatspace:
                # Use supervisor's chatspace from database
                dm_space = {"name": supervisor_chatspace}
                logger.info(f"Using supervisor's chatspace from database: {supervisor_chatspace}")
            else:
                # Fallback: Find any available space
                spaces = chat_service.spaces().list(pageSize=100).execute().get('spaces', [])
                dm_space = None
                
                # Try to find any available space
                for space in spaces:
                    if (
                        isinstance(space, dict) and
                        space.get('spaceType') in ('ROOM', 'SPACE', 'DIRECT_MESSAGE') and
                        space.get('name')
                    ):
                        dm_space = space
                        logger.info(f"Using fallback space: {space.get('name')}")
                        break
                
                if not dm_space:
                    logger.warning(f"No suitable space found for supervisor: {supervisor_email}")
                    return "no_space_available"

            # Build approval URLs
            base_url = os.getenv("APPROVAL_WEBHOOK_URL") or "http://localhost:3005"
            approve_url = f"{base_url}/chat/webhook?applicationId={application_id}&decision=approve" if application_id else "https://example.com/approve"
            deny_url = f"{base_url}/chat/webhook?applicationId={application_id}&decision=deny" if application_id else "https://example.com/deny"

            # Create approval card
            start = leave_data.get('startDate', 'N/A')
            end = leave_data.get('endDate', 'N/A')
            leave_type = leave_data.get('leaveType', 'N/A')
            reason = leave_data.get('reason', 'N/A')
            requester = leave_data.get('requesterEmail', 'Unknown')

            card = {
                "cardsV2": [
                    {
                        "cardId": f"leave-approval-{application_id}",
                        "card": {
                            "header": {
                                "title": "Leave Approval Request", 
                                "subtitle": f"From: {requester}"
                            },
                            "sections": [
                                {
                                    "widgets": [
                                        {"textParagraph": {
                                            "text": f"<b>Leave Type:</b> {leave_type}<br><b>Start Date:</b> {start}<br><b>End Date:</b> {end}<br><b>Reason:</b> {reason}<br><b>Application ID:</b> {application_id}"
                                        }}
                                    ]
                                },
                                {
                                    "header": "Please choose:",
                                    "widgets": [
                                        {"buttonList": {"buttons": [
                                            {"text": "✅ Approve", "onClick": {"openLink": {"url": approve_url}}},
                                            {"text": "❌ Deny", "onClick": {"openLink": {"url": deny_url}}},
                                        ]}}
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }

            chat_service.spaces().messages().create(parent=dm_space['name'], body=card).execute()
            logger.info(f"Supervisor poll sent to {supervisor_email} for application: {application_id}")
            return "sent"
            
        except Exception as error:
            logger.error(f"Supervisor poll send failed: {error}", exc_info=True)
            return "failed"

    def update_approval_status(self, application_id: str, decision: str) -> Optional[Dict[str, Any]]:
        app = self.leave_applications.get(application_id)
        if not app:
            logger.warning(f"Application not found for update: {application_id}")
            return None
        status = "approved" if decision.lower() == "approve" else "denied"
        app["approvalStatus"] = status
        app["status"] = "completed"
        app["decisionAt"] = datetime.now().isoformat()
        self.leave_applications[application_id] = app
        logger.info(f"Approval status updated for application {application_id}: {status}")
        return app
    
    async def process_leave_application(self, leave_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process leave application"""
        try:
            logger.info(f"Processing leave application for: {leave_data.get('requesterEmail')}")
            initial_state: LeaveApplicationState = {
                "leaveData": leave_data,
                "applicationId": None,
                "status": "pending",
                "conflicts": [],
                "approvalStatus": "pending",
                "finalMessage": "",
                "error": None,
            }
            
            # Execute the workflow
            result = await self.workflow.ainvoke(initial_state)
            
            return {
                "success": not result.get("error"),
                "message": result.get("finalMessage", ""),
                "applicationId": result.get("applicationId"),
                "status": result.get("status"),
                "conflicts": result.get("conflicts", []),
                "approvalStatus": result.get("approvalStatus"),
                "error": result.get("error"),
            }
        except Exception as error:
            logger.error(f"Error processing leave application: {error}", exc_info=True)
            return {
                "success": False,
                "message": "Unexpected error while processing your leave request. Please try again shortly.",
                "error": str(error),
            }
    
    def get_application_status(self, application_id: str) -> Optional[Dict[str, Any]]:
        application = self.leave_applications.get(application_id)
        if not application:
            logger.warning(f"Application status requested but not found: {application_id}")
            return None
        logger.info(f"Application status retrieved for: {application_id}")
        return {
            "id": application["id"],
            "status": application.get("status"),
            "approvalStatus": application.get("approvalStatus"),
            "submittedAt": application.get("submittedAt"),
            "approvedAt": application.get("approvedAt"),
            **application,
        }
    
    def get_all_applications(self) -> List[Dict[str, Any]]:
        logger.info("Retrieving all leave applications.")
        return list(self.leave_applications.values())