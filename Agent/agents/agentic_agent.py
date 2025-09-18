import os
import json
from typing import Any, Dict, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # Available in newer langgraph versions
except Exception:
    SqliteSaver = None  # Fallback to in-memory saver below
from langgraph.checkpoint.memory import MemorySaver
from .utils import clean_json_response, normalize_leave_dates


class AgenticState(TypedDict, total=False):
    userEmail: str
    lastUserMessage: str
    leaveData: Dict[str, Any]
    route: str
    finalMessage: str
    error: Optional[str]


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

        self.graph = self._build_graph()

    def _state_reducer(self):
        return {
            "userEmail": lambda x, y: y if y is not None else x,
            "lastUserMessage": lambda x, y: y if y is not None else x,
            "leaveData": lambda x, y: y if y is not None else x or {},
            "route": lambda x, y: y if y is not None else x,
            "finalMessage": lambda x, y: y if y is not None else x,
            "error": lambda x, y: y if y is not None else x,
        }

    async def _decide_next(self, state: AgenticState) -> AgenticState:
        try:
            system = (
                "You are an autonomous flow controller. Decide the next step from the user's latest message and partial leave data.\n\n"
                "Choose one route strictly:\n"
                "- collect_leave: user intent about leave; we need to extract or update fields.\n"
                "- run_leave_workflow: all REQUIRED fields present (startDate, endDate, leaveType, reason, supervisorEmail).\n"
                "- general_chat: otherwise.\n\n"
                "Return only JSON: {\"route\": \"collect_leave|run_leave_workflow|general_chat\"}"
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
            system = (
                "Extract or update leave fields from the user's latest message.\n"
                "Fields: startDate, endDate, leaveType, reason, supervisorEmail.\n"
                "Return ONLY JSON with any fields found. Do not include explanations."
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

            updated = {**current, **{k: v for k, v in updates.items() if v}}
            return {**state, "leaveData": updated}
        except Exception as error:
            return {**state, "error": str(error), "finalMessage": "Sorry—couldn’t parse your leave details."}

    async def _validate_leave(self, state: AgenticState) -> AgenticState:
        try:
            leave = state.get("leaveData", {})
            normalized, err = normalize_leave_dates(leave)
            if err:
                return {**state, "leaveData": normalized, "finalMessage": err}
            required = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
            missing = [f for f in required if not normalized.get(f)]
            if missing:
                return {**state, "leaveData": normalized, "finalMessage": ", ".join(missing)}
            return {**state, "leaveData": normalized, "route": "run_leave_workflow"}
        except Exception as error:
            return {**state, "error": str(error), "finalMessage": "Validation failed. Please check your inputs."}

    async def _craft_prompt(self, state: AgenticState) -> AgenticState:
        try:
            leave = state.get("leaveData", {})
            required = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
            missing = [f for f in required if not leave.get(f)]
            system = (
                "You are a helpful assistant. Ask ONE concise question to collect the next most critical missing field. "
                "Use examples if the field is a date. Keep under 25 words."
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps({
                    "latest": state.get("lastUserMessage", ""),
                    "missing": missing,
                    "have": {k: v for k, v in leave.items() if v},
                })),
            ]
            resp = await self.llm.ainvoke(messages)
            return {**state, "finalMessage": resp.content}
        except Exception as error:
            return {**state, "error": str(error), "finalMessage": "Please share the next missing detail."}

    async def _general_chat(self, state: AgenticState) -> AgenticState:
        try:
            system = (
                "You are a concise and helpful workplace assistant. Answer briefly and helpfully."
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=state.get("lastUserMessage", "")),
            ]
            resp = await self.llm.ainvoke(messages)
            return {**state, "finalMessage": resp.content}
        except Exception as error:
            return {**state, "error": str(error), "finalMessage": "I'm having trouble responding right now."}

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

    async def process_message(self, user_message: str, user_email: str) -> Dict[str, Any]:
        initial: AgenticState = {
            "userEmail": user_email,
            "lastUserMessage": user_message,
        }

        thread_id = self.user_thread_ids.get(user_email, user_email)

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


