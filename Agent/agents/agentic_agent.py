import os
import json
from typing import Any, Dict, Optional, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver


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

        memory_path = os.getenv("LANGGRAPH_SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "agent_memory.sqlite3"))
        self.checkpointer = SqliteSaver.from_conn_string(os.path.abspath(memory_path))

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
                "You are a routing controller for a workplace assistant. "
                "Decide the next step based on the user's latest message and any partial leave data.\n\n"
                "ROUTES:\n"
                "- collect_leave: if the user is providing or asking about leave request details.\n"
                "- run_leave_workflow: if all required leave fields are present (startDate, endDate, leaveType, reason, supervisorEmail).\n"
                "- general_chat: for everything else.\n\n"
                "Return ONLY a compact JSON: {\"route\": one_of(collect_leave, run_leave_workflow, general_chat)}."
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=json.dumps({
                    "message": state.get("lastUserMessage", ""),
                    "leaveData": state.get("leaveData", {}),
                })),
            ]
            resp = await self.llm.ainvoke(messages)
            content = resp.content.strip()
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
            content = resp.content.strip()
            # best-effort JSON cleanup
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            try:
                updates = json.loads(content)
                if not isinstance(updates, dict):
                    updates = {}
            except Exception:
                updates = {}

            updated = {**current, **{k: v for k, v in updates.items() if v}}
            missing = [f for f in ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"] if not updated.get(f)]
            if missing:
                prompt = (
                    "I recorded your details. Please provide the following to proceed: "
                    + ", ".join(missing)
                )
                return {**state, "leaveData": updated, "finalMessage": prompt}
            return {**state, "leaveData": updated, "route": "run_leave_workflow"}
        except Exception as error:
            return {**state, "error": str(error), "finalMessage": "Sorry—couldn’t parse your leave details."}

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
                "run_leave_workflow": "run_leave_workflow",
            },
        )

        # collect_leave can either finish or go run workflow next time
        workflow.add_edge("collect_leave", END)
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


