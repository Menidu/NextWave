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
        }

    async def _decide_next(self, state: AgenticState) -> AgenticState:
        try:
            system = (
                "You orchestrate a friendly workplace assistant. Decide the next step from the latest user message and partial leave data.\n\n"
                "ROUTE RULES (pick exactly one):\n"
                "- collect_leave: user is talking about time off/leave OR providing details; we should extract/update fields.\n"
                "- run_leave_workflow: ALL REQUIRED fields present (startDate, endDate, leaveType, reason, supervisorEmail).\n"
                "- general_chat: everything else (answer helpfully).\n\n"
                "Respond ONLY with JSON: {\"route\": \"collect_leave|run_leave_workflow|general_chat\"}"
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps({
                    "message": state.get("lastUserMessage", ""),
                    "leaveData": state.get("leaveData", {}),
                })),
            ]
            resp = await self.llm.ainvoke(messages)
            content = clean_json_response(resp.content)
            try:
                data = json.loads(content)
                route = data.get("route")
                if route not in {"collect_leave", "run_leave_workflow", "general_chat"}:
                    route = "general_chat"
            except Exception:
                route = "general_chat"
            return {**state, "route": route}
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
            # Get intelligent policy context for the query using LLM
            user_query = state.get("lastUserMessage", "")
            policy_context = await self.policy_context.get_intelligent_policy_context(user_query, self.llm)
            
            # Create Mira's persona
            display_name = state.get("userDisplayName", "")
            first_name = display_name.split()[0] if display_name else "there"
            is_first = state.get("isFirstMessage", False)
            
            greeting = f"Hello {first_name}! I'm Mira, your workplace assistant. " if is_first else ""
            
            today_iso = datetime.now(timezone.utc).date().isoformat()
            system = f"""You are Mira, a friendly and helpful workplace assistant. You have access to company policies and can help with various workplace questions.

{greeting}Be professional, supportive, and concise. Offer actionable help and short examples when useful.

Today's date: {today_iso}. When the user mentions relative dates (e.g., "next Monday"), interpret them relative to today's date.

RELEVANT COMPANY POLICIES:
{policy_context if policy_context else "No policy documents are currently available."}

Instructions for policy questions:
- When users ask about workplace policies, intelligently select and reference the relevant policy sections from the context above
- Provide specific, actionable information based on the policies
- If the user's question relates to multiple policies, reference all relevant sections
- If you don't find specific information in the policies, be honest about it and suggest they contact HR
- Always cite which policy section you're referencing (e.g., "According to our Leave Policy...")

Always maintain a warm, helpful tone while being professional and accurate. Respond in plain text only (no markdown, no bullets)."""
            
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=state.get("lastUserMessage", "")),
            ]
            resp = await self.llm.ainvoke(messages)
            return {**state, "finalMessage": resp.content}
        except Exception as error:
            msg = await self._llm_safe_message("Provide a brief, friendly apology and ask how to help further.")
            return {**state, "error": str(error), "finalMessage": msg}

    def _build_graph(self):
        workflow = StateGraph(AgenticState)
        workflow.add_node("decide_next", self._decide_next)
        workflow.add_node("collect_leave", self._collect_leave_fields)
        workflow.add_node("validate_leave", self._validate_leave)
        workflow.add_node("craft_prompt", self._craft_prompt)
        workflow.add_node("general_chat", self._general_chat)

        # Entry and routing
        workflow.set_entry_point("decide_next")

        def route_selector(state: AgenticState) -> str:
            route = state.get("route")
            if route == "collect_leave":
                return "collect_leave"
            if route == "run_leave_workflow":
                return "run_leave_workflow"
            return "general_chat"

        workflow.add_conditional_edges(
            "decide_next",
            route_selector,
            {
                "collect_leave": "collect_leave",
                "general_chat": "general_chat",
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
        
        initial: AgenticState = {
            "userEmail": user_email,
            "userDisplayName": user_display_name,
            "lastUserMessage": user_message,
            "isFirstMessage": is_first_message,
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


