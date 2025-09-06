import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


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
        
        # Build the LangGraph workflow
        self.workflow = self.build_workflow()
    
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
                return {
                    **state,
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                    "finalMessage": "I'm sorry, but some required information is missing from your leave application. Please try again.",
                    "status": "failed",
                }
            
            # Create leave application record
            application = {
                "id": application_id,
                **leave_data,
                "submittedAt": datetime.now().isoformat(),
                "status": "pending",
            }
            
            self.leave_applications[application_id] = application
            
            return {
                **state,
                "applicationId": application_id,
                "status": "created",
            }
        except Exception as error:
            print(f"Error in validate_and_create_application: {error}")
            return {
                **state,
                "error": str(error),
                "finalMessage": "An error occurred while processing your leave application. Please try again.",
                "status": "failed",
            }
    
    async def check_calendar_conflicts(self, state: LeaveApplicationState) -> LeaveApplicationState:
        """Step 2: Check for calendar conflicts"""
        try:
            leave_data = state["leaveData"]
            
            # Simulate calendar conflict checking
            conflicts = await self.simulate_calendar_check(leave_data)
            
            return {
                **state,
                "conflicts": conflicts,
                "status": "conflicts_found" if conflicts else "no_conflicts",
            }
        except Exception as error:
            print(f"Error in check_calendar_conflicts: {error}")
            return {
                **state,
                "error": str(error),
                "finalMessage": "An error occurred while checking for calendar conflicts.",
                "status": "failed",
            }
    
    async def process_approval_workflow(self, state: LeaveApplicationState) -> LeaveApplicationState:
        """Step 3: Process approval workflow"""
        try:
            leave_data = state["leaveData"]
            conflicts = state["conflicts"]
            application_id = state["applicationId"]
            
            if conflicts:
                return {
                    **state,
                    "approvalStatus": "requires_manual_review",
                    "finalMessage": f"""⚠️ **Calendar Conflicts Detected**

Your leave application has been submitted, but there are some calendar conflicts that need manual review:

{chr(10).join(f'- {conflict}' for conflict in conflicts)}

Your application ID is: `{application_id}`

A manager will review your request and get back to you within 24 hours.""",
                    "status": "completed",
                }
            
            # Simulate approval process
            approval_result = await self.simulate_approval_process(leave_data)
            
            # Update application status
            if application_id and application_id in self.leave_applications:
                application = self.leave_applications[application_id]
                application["approvalStatus"] = approval_result["status"]
                application["approvedAt"] = datetime.now().isoformat()
                self.leave_applications[application_id] = application
            
            return {
                **state,
                "approvalStatus": approval_result["status"],
                "finalMessage": approval_result["message"],
                "status": "completed",
            }
        except Exception as error:
            print(f"Error in process_approval_workflow: {error}")
            return {
                **state,
                "error": str(error),
                "finalMessage": "An error occurred during the approval process.",
                "status": "failed",
            }
    
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
        
        except Exception as error:
            print(f"Error in simulate_calendar_check: {error}")
        
        return conflicts
    
    async def simulate_approval_process(self, leave_data: Dict[str, Any]) -> Dict[str, str]:
        """Simulate approval process"""
        try:
            start_date = datetime.strptime(leave_data["startDate"], "%Y-%m-%d") if leave_data["startDate"].count("-") == 2 else datetime.now()
            end_date = datetime.strptime(leave_data["endDate"], "%Y-%m-%d") if leave_data["endDate"].count("-") == 2 else datetime.now()
            duration = max(1, (end_date - start_date).days + 1)
            
            # Auto-approve short casual leaves
            if "casual" in leave_data.get("leaveType", "").lower() and duration <= 2:
                return {
                    "status": "approved",
                    "message": f"""✅ **Leave Application Approved!**

📅 **Leave Details:**
- **Start Date:** {leave_data['startDate']}
- **End Date:** {leave_data['endDate']}
- **Leave Type:** {leave_data['leaveType']}
- **Duration:** {duration} day(s)
- **Reason:** {leave_data['reason']}

Your leave has been automatically approved. Enjoy your time off!""",
                }
            
            # Auto-approve sick leave
            if "sick" in leave_data.get("leaveType", "").lower():
                return {
                    "status": "approved",
                    "message": f"""✅ **Sick Leave Approved!**

📅 **Leave Details:**
- **Start Date:** {leave_data['startDate']}
- **End Date:** {leave_data['endDate']}
- **Leave Type:** {leave_data['leaveType']}
- **Duration:** {duration} day(s)
- **Reason:** {leave_data['reason']}

Your sick leave has been approved. Take care and get well soon!""",
                }
            
            # Require manual approval for other types
            return {
                "status": "pending_approval",
                "message": f"""⏳ **Leave Application Submitted**

📅 **Leave Details:**
- **Start Date:** {leave_data['startDate']}
- **End Date:** {leave_data['endDate']}
- **Leave Type:** {leave_data['leaveType']}
- **Duration:** {duration} day(s)
- **Reason:** {leave_data['reason']}

Your leave application has been submitted and is pending manager approval. You'll receive a notification once it's reviewed.""",
            }
        
        except Exception as error:
            print(f"Error in simulate_approval_process: {error}")
            return {
                "status": "error",
                "message": "An error occurred during approval simulation.",
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
    
    async def process_leave_application(self, leave_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to process leave application"""
        try:
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
            print(f"Error processing leave application: {error}")
            return {
                "success": False,
                "message": "An unexpected error occurred while processing your leave application. Please try again.",
                "error": str(error),
            }
    
    def get_application_status(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get application status"""
        application = self.leave_applications.get(application_id)
        if not application:
            return None
        
        return {
            "id": application["id"],
            "status": application.get("status"),
            "approvalStatus": application.get("approvalStatus"),
            "submittedAt": application.get("submittedAt"),
            "approvedAt": application.get("approvedAt"),
            **application,
        }
    
    def get_all_applications(self) -> List[Dict[str, Any]]:
        """List all applications (for admin purposes)"""
        return list(self.leave_applications.values())