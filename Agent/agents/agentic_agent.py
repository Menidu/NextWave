import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TypedDict
# Using Google Gemini via LangChain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # Available in newer langgraph versions
except Exception:
    SqliteSaver = None  # Fallback to in-memory saver below
from langgraph.checkpoint.memory import MemorySaver
from .utils import clean_json_response, normalize_leave_dates
from .policy_context import PolicyContext
from .rm_agent import RMAgent

#gee
class AgenticState(TypedDict, total=False):
    userEmail: str
    userDisplayName: str
    lastUserMessage: str
    leaveData: Dict[str, Any]
    route: str
    finalMessage: str
    error: Optional[str]
    missingFields: str
    validationError: str
    isFirstMessage: bool
    userContext: Dict[str, Any]


class AgenticAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )

        # Prefer SQLite persistence when available; otherwise fall back to in-memory saver
        if SqliteSaver is not None:
            memory_path = os.getenv("LANGGRAPH_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "agent_memory.sqlite3"))
            self.checkpointer = SqliteSaver.from_conn_string(os.path.abspath(memory_path))
        else:
            self.checkpointer = MemorySaver()

        # Runtime mapping to rotate thread IDs per user when resetting memory
        self.user_thread_ids: Dict[str, str] = {}
        
        # Initialize policy context
        self.policy_context = PolicyContext(os.path.join(os.path.dirname(__file__), "..", "docs"))
        
        # Initialize RM Agent
        self.rm_agent = RMAgent()

        self.graph = self._build_graph()

    def _state_reducer(self):
        return {
            "userEmail": lambda x, y: y if y is not None else x,
            "userDisplayName": lambda x, y: y if y is not None else x,
            "lastUserMessage": lambda x, y: y if y is not None else x,
            "leaveData": lambda x, y: y if y is not None else x or {},
            "route": lambda x, y: y if y is not None else x,
            "finalMessage": lambda x, y: y if y is not None else x,
            "error": lambda x, y: y if y is not None else x,
            "missingFields": lambda x, y: y if y is not None else x,
            "validationError": lambda x, y: y if y is not None else x,
            "isFirstMessage": lambda x, y: y if y is not None else x,
            "userContext": lambda x, y: y if y is not None else x,
        }

    async def _decide_next(self, state: AgenticState) -> AgenticState:
        try:
            user_message = state.get('lastUserMessage', '')
            leave_data = state.get("leaveData", {})
            
            # Simple keyword-based routing for reliability
            message_lower = user_message.lower()
            
            # Check for RM-related keywords first
            rm_keywords = [
                "supervisor", "department", "employee", "team", "role", "employee id", 
                "hire", "manager", "hr", "organization", "staff", "colleague", "my info",
                "who am i", "what am i", "employee details", "personal info"
            ]
            
            if any(keyword in message_lower for keyword in rm_keywords):
                return {**state, "route": "rm_query"}
            
            # Check for leave-related keywords
            leave_keywords = [
                "leave", "vacation", "sick", "time off", "absence", "holiday", 
                "day off", "personal day", "maternity", "paternity", "emergency leave"
            ]
            
            if any(keyword in message_lower for keyword in leave_keywords):
                # Check if we have all required fields
                required_fields = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
                has_all_fields = all(leave_data.get(field) for field in required_fields)
                
                if has_all_fields:
                    return {**state, "route": "run_leave_workflow"}
                else:
                    return {**state, "route": "collect_leave"}
            
            # Everything else goes to general chat
            return {**state, "route": "general_chat"}
            
        except Exception as error:
            return {**state, "route": "general_chat", "error": str(error)}

    async def _collect_leave_fields(self, state: AgenticState) -> AgenticState:
        try:
            today_iso = datetime.now(timezone.utc).date().isoformat()
            system = (
                f"You extract leave details from a friendly workplace chat.\n" , 
                "your job is to extract any leave-related fields from the user's message.\n Interact naturally and helpfully.\n and dont repeat yourself when requesting details again "
                f"Today is {today_iso}. Interpret relative dates (e.g., 'next Monday') relative to today.\n"
                "Fields to capture: startDate, endDate, leaveType, reason, supervisorEmail.\n"
                "For supervisorEmail: extract any email address mentioned, even if not explicitly labeled.\n"
                "Return ONLY compact JSON with fields found (no prose)."
            )
            current = state.get("leaveData", {})
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=f"message: {state.get('lastUserMessage','')}\ncurrent: {json.dumps(current)}"),
            ]
            resp = await self.llm.ainvoke(messages)
            content = clean_json_response(resp.content)
            try:
                updates = json.loads(content)
                if not isinstance(updates, dict):
                    updates = {}
            except Exception:
                updates = {}

            # Extract email from message if not in updates
            if not updates.get("supervisorEmail") and current.get("supervisorEmail"):
                from .utils import extract_email
                email = extract_email(state.get("lastUserMessage", ""))
                if email:
                    updates["supervisorEmail"] = email

            updated = {**current, **{k: v for k, v in updates.items() if v}}
            return {**state, "leaveData": updated}
        except Exception as error:
            # Ask LLM to inform the user about parsing trouble
            msg = await self._llm_safe_message("We had trouble parsing leave details from the last message. Ask for one specific missing detail.")
            return {**state, "error": str(error), "finalMessage": msg}

    async def _validate_leave(self, state: AgenticState) -> AgenticState:
        try:
            leave = state.get("leaveData", {})
            normalized, err = normalize_leave_dates(leave)
            if err:
                # Route to crafted prompt with validation context
                return {**state, "leaveData": normalized, "validationError": err}
            
            required = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
            missing = [f for f in required if not normalized.get(f)]
            
            if missing:
                # Special handling for supervisor email - ask directly
                if "supervisorEmail" in missing:
                    return {**state, "leaveData": normalized, "finalMessage": "I need your supervisor's email address to send the approval request. Please provide it (e.g., john@company.com)."}
                
                # Route to crafted prompt with missing fields context
                return {**state, "leaveData": normalized, "missingFields": missing}
            
            return {**state, "leaveData": normalized, "route": "run_leave_workflow"}
        except Exception as error:
            msg = await self._llm_safe_message("Validation failed for leave inputs. Ask for the most critical missing/corrected field politely.")
            return {**state, "error": str(error), "finalMessage": msg}

    async def _craft_prompt(self, state: AgenticState) -> AgenticState:
        try:
            leave = state.get("leaveData", {})
            required = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
            missing = state.get("missingFields") or [f for f in required if not leave.get(f)]
            
            # Special handling for supervisor email
            if "supervisorEmail" in missing:
                return {**state, "finalMessage": "I need your supervisor's email address to send the approval request. Please provide it (e.g., john@company.com)."}
            
            today_iso = datetime.now(timezone.utc).date().isoformat()
            system = (
                f"You are a friendly workplace assistant. Today is {today_iso}. "
                "Ask ONE concise, polite question to collect the next most important missing field. "
                "Priority order: startDate, endDate, leaveType, reason. "
                "If asking for a date, give a quick example (e.g., 2025-01-15 or 'next Monday'). Keep under 25 words. "
                "Respond in plain text only (no markdown, no bullets)."
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps({
                    "latest": state.get("lastUserMessage", ""),
                    "missing": missing,
                    "validationError": state.get("validationError"),
                    "have": {k: v for k, v in leave.items() if v},
                })),
            ]
            resp = await self.llm.ainvoke(messages)
            return {**state, "finalMessage": resp.content}
        except Exception as error:
            msg = await self._llm_safe_message("Ask for the next missing leave detail in under 25 words.")
            return {**state, "error": str(error), "finalMessage": msg}

    async def _general_chat(self, state: AgenticState) -> AgenticState:
        try:
            user_query = state.get("lastUserMessage", "")
            user_context = state.get("userContext", {})
            display_name = state.get("userDisplayName", "")
            first_name = display_name.split()[0] if display_name else "there"
            is_first = state.get("isFirstMessage", False)
            
            greeting = f"Hello {first_name}! I'm Mira, your workplace assistant. " if is_first else ""
            
            # Check if this might be an RM query that wasn't caught by routing
            message_lower = user_query.lower()
            rm_hints = ["my", "i am", "i work", "my team", "my department"]
            
            if any(hint in message_lower for hint in rm_hints) and not user_context:
                # Try to get user context for potential RM query
                user_email = state.get("userEmail", "")
                if user_email:
                    user_context = await self.rm_agent.get_user_data(user_email)
                    if user_context:
                        # This is actually an RM query, handle it
                        return await self._rm_query({**state, "userContext": user_context})
            
            # Get policy context
            try:
                policy_context = await self.policy_context.get_intelligent_policy_context(user_query, self.llm)
            except Exception:
                policy_context = "No policy documents are currently available."
            
            today_iso = datetime.now(timezone.utc).date().isoformat()
            
            # Create a helpful response based on query type
            if "policy" in message_lower or "dress code" in message_lower or "safety" in message_lower:
                response = f"{greeting}I can help you with workplace policies. {policy_context[:200]}... For more specific information, please contact HR."
            elif "help" in message_lower or "what can you do" in message_lower:
                response = f"{greeting}I can help you with:\n- Leave applications and time off requests\n- Workplace policies and procedures\n- Employee information (if you're in our database)\n- General workplace questions\n\nWhat would you like to know?"
            elif "hello" in message_lower or "hi" in message_lower:
                response = f"{greeting}How can I help you today? I can assist with leave requests, workplace policies, or answer general questions."
            else:
                response = f"{greeting}I'm here to help with workplace questions, leave requests, and policies. Could you be more specific about what you need assistance with?"
            
            return {**state, "finalMessage": response}
            
        except Exception as error:
            msg = f"Hello! I'm Mira, your workplace assistant. I'm having trouble processing your request right now. Please try again or contact HR for assistance."
            return {**state, "error": str(error), "finalMessage": msg}

    async def _rm_query(self, state: AgenticState) -> AgenticState:
        """Handle RM-related queries using RM Agent as a tool"""
        try:
            user_email = state.get("userEmail", "")
            user_message = state.get("lastUserMessage", "")
            user_context = state.get("userContext", {})
            
            # Get user data if not already available
            if not user_context:
                user_context = await self.rm_agent.get_user_data(user_email)
            
            # Create a more helpful response using the RM Agent as a tool
            if user_context:
                # User found in database - provide personalized response
                name = user_context.get('name', 'there')
                department = user_context.get('department', 'your department')
                role = user_context.get('role', 'your role')
                supervisor = user_context.get('supervisor_name', 'your supervisor')
                employee_id = user_context.get('employee_id', 'your employee ID')
                
                # Generate response based on query type
                message_lower = user_message.lower()
                
                if "supervisor" in message_lower:
                    response = f"Hi {name}! Your supervisor is {supervisor} ({user_context.get('supervisor_email', '')})."
                elif "department" in message_lower:
                    response = f"Hi {name}! You work in the {department} department."
                elif "role" in message_lower or "job" in message_lower:
                    response = f"Hi {name}! Your role is {role} in the {department} department."
                elif "employee id" in message_lower or "id" in message_lower:
                    response = f"Hi {name}! Your employee ID is {employee_id}."
                elif "hire" in message_lower or "join" in message_lower:
                    hire_date = user_context.get('hire_date', 'your hire date')
                    response = f"Hi {name}! You joined the company on {hire_date}."
                else:
                    response = f"Hi {name}! Here's your information: You're a {role} in the {department} department. Your supervisor is {supervisor} and your employee ID is {employee_id}."
            else:
                # User not found - provide general assistance
                response = f"I don't have your specific information in the database. Please contact HR for your employee details, or you can ask me about general workplace policies and procedures."
            
            return {**state, "finalMessage": response}
            
        except Exception as error:
            msg = f"I encountered an error while looking up your information. Please try again or contact HR for assistance."
            return {**state, "error": str(error), "finalMessage": msg}

    def _build_graph(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("decide_next", self._decide_next)
        workflow.add_node("collect_leave", self._collect_leave_fields)
        workflow.add_node("validate_leave", self._validate_leave)
        workflow.add_node("craft_prompt", self._craft_prompt)
        workflow.add_node("general_chat", self._general_chat)
        workflow.add_node("rm_query", self._rm_query)

        # Entry and routing
        workflow.set_entry_point("decide_next")

        def route_selector(state: AgenticState) -> str:
            route = state.get("route")
            if route == "collect_leave":
                return "collect_leave"
            if route == "run_leave_workflow":
                return "run_leave_workflow"
            if route == "rm_query":
                return "rm_query"
            return "general_chat"

        workflow.add_conditional_edges(
            "decide_next",
            route_selector,
            {
                "collect_leave": "collect_leave",
                "general_chat": "general_chat",
                "rm_query": "rm_query",
                "run_leave_workflow": END,
            },
        )

        # After collecting, validate; after validation, either workflow or craft a question
        workflow.add_edge("collect_leave", "validate_leave")
        def post_validate(state: AgenticState) -> str:
            if state.get("route") == "run_leave_workflow":
                return "run_leave_workflow"
            # If we have a finalMessage with missing guidance, ask crafted prompt
            return "craft_prompt"
        workflow.add_conditional_edges(
            "validate_leave",
            post_validate,
            {
                "run_leave_workflow": END,
                "craft_prompt": "craft_prompt",
            },
        )

        # collect_leave can either finish or go run workflow next time
        workflow.add_edge("craft_prompt", END)
        workflow.add_edge("general_chat", END)

        # run_leave_workflow is provided by external manager; we end here and let caller invoke downstream
        # The manager will detect the route and call LeaveAgent.

        return workflow.compile(checkpointer=self.checkpointer)

    async def process_message(self, user_message: str, user_email: str, user_display_name: str = "") -> Dict[str, Any]:
        # Check if this is the first message from this user
        thread_id = self.user_thread_ids.get(user_email, user_email)
        is_first_message = user_email not in self.user_thread_ids
        
        # Retrieve user context on first message
        user_context = {}
        if is_first_message:
            user_context = await self.rm_agent.get_user_data(user_email) or {}
        
        initial: AgenticState = {
            "userEmail": user_email,
            "userDisplayName": user_display_name,
            "lastUserMessage": user_message,
            "isFirstMessage": is_first_message,
            "userContext": user_context,
        }

        result = await self.graph.ainvoke(
            initial,
            config={"configurable": {"thread_id": thread_id}},
        )

        response = {
            "response": result.get("finalMessage") or "",
            "leaveData": result.get("leaveData") or {},
            "route": result.get("route"),
            "error": result.get("error"),
        }
        return response

    def reset_thread(self, user_email: str) -> None:
        # Generate a new logical thread id for this user for subsequent interactions
        import uuid
        self.user_thread_ids[user_email] = f"{user_email}::{uuid.uuid4()}"

    async def _llm_safe_message(self, instruction: str) -> str:
        try:
            system = "You are a helpful assistant. Follow the instruction to produce a single short user-facing message."
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=instruction),
            ]
            resp = await self.llm.ainvoke(messages)
            return resp.content or ""
        except Exception:
            # Last resort minimal message
            return "Sorry, something went wrong. Could you rephrase or provide the next detail?"


