from typing import TypedDict, Optional, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from enum import Enum
import json
import uuid
from datetime import datetime, timezone

# =========================================================
# Session Management
# =========================================================

class LeaveSession:
    def __init__(self, user_id: str, session_id: str = None):
        self.session_id = session_id or f"leave_{uuid.uuid4().hex[:8]}"
        self.user_id = user_id
        self.leave_request_id = None
        self.slots = {
            "start_date": None,
            "end_date": None,
            "type": None,
            "reason": None
        }
        self.current_stage = "start"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_updated = datetime.now(timezone.utc).isoformat()
    
    def update(self):
        self.last_updated = datetime.now(timezone.utc).isoformat()

class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: str) -> LeaveSession:
        # End any existing session for this user
        if user_id in self.sessions:
            del self.sessions[user_id]
        
        session = LeaveSession(user_id)
        self.sessions[user_id] = session
        return session
    
    def get_session(self, user_id: str) -> Optional[LeaveSession]:
        return self.sessions.get(user_id)
    
    def end_session(self, user_id: str):
        if user_id in self.sessions:
            del self.sessions[user_id]

session_manager = SessionManager()

# =========================================================
# Enhanced State Definition with Session Support
# =========================================================

class AgentState(TypedDict):
    user_id: str
    user_message: str
    bot_response: str
    intent: Optional[str]
    next_node: Optional[str]
    structured_data: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    session_id: Optional[str]
    current_stage: Optional[str]
    should_continue_session: bool

# =========================================================
# Enhanced AI Intent Classifier with Session Awareness
# =========================================================

def ai_intent_classifier_node(state: AgentState) -> AgentState:
    """AI-powered intent classification with session awareness"""
    user_message = state["user_message"]
    user_id = state["user_id"]
    
    # Check if user has an active leave session
    active_session = session_manager.get_session(user_id)
    if active_session:
        state["session_id"] = active_session.session_id
        state["current_stage"] = active_session.current_stage
        state["next_node"] = "leave_agent"
        state["intent"] = "leave_request_continue"
        state["should_continue_session"] = True
        return state
    
    # Simplified intent classification for Google Chat integration
    user_message_lower = user_message.lower()
    
    if any(word in user_message_lower for word in ["leave", "time off", "vacation", "holiday"]):
        state["intent"] = "leave_request"
        state["next_node"] = "leave_agent"
        state["should_continue_session"] = True
        state["bot_response"] = "I'd be happy to help you apply for leave."
    elif any(word in user_message_lower for word in ["hi", "hello", "hey", "greetings"]):
        state["intent"] = "greeting"
        state["next_node"] = "greeting_response"
        state["should_continue_session"] = False
    else:
        state["intent"] = "general_qa"
        state["next_node"] = "rag_agent"
        state["should_continue_session"] = False
    
    return state

# =========================================================
# Enhanced Leave Agent with Session Persistence
# =========================================================

def ai_leave_agent_node(state: AgentState) -> AgentState:
    """AI-powered Leave Agent with session persistence"""
    user_id = state["user_id"]
    user_message = state["user_message"]
    
    # Get or create session
    session = session_manager.get_session(user_id)
    if not session:
        session = session_manager.create_session(user_id)
    
    session.update()
    state["session_id"] = session.session_id
    state["current_stage"] = session.current_stage
    
    # Route based on current stage
    if session.current_stage == "start":
        return start_leave_process(state, session)
    elif session.current_stage == "collecting_dates":
        return collect_dates(state, session)
    elif session.current_stage == "collecting_type":
        return collect_leave_type(state, session)
    elif session.current_stage == "collecting_reason":
        return collect_reason(state, session)
    elif session.current_stage == "confirmation":
        return confirm_leave_request(state, session)
    else:
        state["bot_response"] = "I'm not sure what to do next. Let's start over."
        session_manager.end_session(user_id)
        state["next_node"] = "end"
        state["should_continue_session"] = False
        return state

def start_leave_process(state: AgentState, session: LeaveSession) -> AgentState:
    """Start a new leave request process"""
    # Extract any initial information from user message
    extracted_data = extract_leave_info(state["user_message"], session.slots)
    session.slots.update(extracted_data)
    
    # Determine next stage based on what's missing
    if not session.slots["start_date"]:
        session.current_stage = "collecting_dates"
        state["bot_response"] = "I'd be happy to help you apply for leave. When would you like to take time off? Please provide start and end dates (e.g., 'next Monday to Friday')."
    elif not session.slots["type"]:
        session.current_stage = "collecting_type"
        state["bot_response"] = "Great! What type of leave would you like to take? (annual, sick, personal, etc.)"
    elif not session.slots["reason"]:
        session.current_stage = "collecting_reason"
        state["bot_response"] = "Could you please tell me the reason for your leave?"
    else:
        session.current_stage = "confirmation"
        state["bot_response"] = generate_confirmation_message(session.slots)
    
    state["next_node"] = "leave_agent"
    state["should_continue_session"] = True
    return state

def collect_dates(state: AgentState, session: LeaveSession) -> AgentState:
    """Collect date information"""
    extracted_data = extract_leave_info(state["user_message"], session.slots)
    session.slots.update(extracted_data)
    
    if session.slots["start_date"] and session.slots["end_date"]:
        session.current_stage = "collecting_type"
        state["bot_response"] = f"Got it! {session.slots['start_date']} to {session.slots['end_date']}. What type of leave would you like to take?"
    else:
        state["bot_response"] = "I need to know the dates for your leave. Please provide start and end dates (e.g., 'next Monday to Friday')."
    
    state["next_node"] = "leave_agent"
    state["should_continue_session"] = True
    return state

def collect_leave_type(state: AgentState, session: LeaveSession) -> AgentState:
    """Collect leave type information"""
    extracted_data = extract_leave_info(state["user_message"], session.slots)
    session.slots.update(extracted_data)
    
    if session.slots["type"]:
        session.current_stage = "collecting_reason"
        state["bot_response"] = f"{session.slots['type'].title()} leave, noted! Could you please tell me the reason?"
    else:
        state["bot_response"] = "What type of leave would you like to take? (annual, sick, personal, emergency, vacation)"
    
    state["next_node"] = "leave_agent"
    state["should_continue_session"] = True
    return state

def collect_reason(state: AgentState, session: LeaveSession) -> AgentState:
    """Collect reason information"""
    extracted_data = extract_leave_info(state["user_message"], session.slots)
    session.slots.update(extracted_data)
    
    if session.slots["reason"]:
        session.current_stage = "confirmation"
        state["bot_response"] = generate_confirmation_message(session.slots)
    else:
        state["bot_response"] = "Could you please provide a reason for your leave?"
    
    state["next_node"] = "leave_agent"
    state["should_continue_session"] = True
    return state

def confirm_leave_request(state: AgentState, session: LeaveSession) -> AgentState:
    """Final confirmation and create leave request"""
    user_id = state["user_id"]
    
    # In a real implementation, you would create the actual leave request in database
    # For now, we'll just simulate it
    
    num_days = calculate_leave_days(session.slots["start_date"], session.slots["end_date"])
    state["bot_response"] = f"✅ Leave request created! \n\n• Dates: {session.slots['start_date']} to {session.slots['end_date']} ({num_days} days)\n• Type: {session.slots['type']}\n• Reason: {session.slots['reason']}\n\nI'll now check for calendar conflicts and send it for approval."
    
    # End the session
    session_manager.end_session(user_id)
    state["next_node"] = "end"
    state["should_continue_session"] = False
    return state

def extract_leave_info(user_message: str, current_slots: Dict) -> Dict:
    """Extract leave information using simple pattern matching"""
    # Simplified extraction for demo purposes
    # In a real implementation, you would use more sophisticated NLP
    
    result = {}
    user_message_lower = user_message.lower()
    
    # Check for date patterns (simplified)
    if "to" in user_message and ("monday" in user_message_lower or "tuesday" in user_message_lower or 
                                "wednesday" in user_message_lower or "thursday" in user_message_lower or 
                                "friday" in user_message_lower or "saturday" in user_message_lower or 
                                "sunday" in user_message_lower):
        parts = user_message.split("to")
        if len(parts) >= 2:
            result["start_date"] = parts[0].strip()
            result["end_date"] = parts[1].strip()
    
    # Check for leave types
    leave_types = ["annual", "sick", "personal", "emergency", "vacation"]
    for leave_type in leave_types:
        if leave_type in user_message_lower:
            result["type"] = leave_type
            break
    
    # If no specific reason detected, use the whole message as reason
    if not result.get("reason") and len(user_message) > 10:
        result["reason"] = user_message
    
    return result

def calculate_leave_days(start_date: str, end_date: str) -> int:
    """Calculate number of leave days (simplified)"""
    # In a real implementation, you would parse dates and calculate difference
    return 5  # Default to 5 days for demo

def generate_confirmation_message(slots: Dict) -> str:
    """Generate confirmation message with collected information"""
    num_days = calculate_leave_days(slots["start_date"], slots["end_date"])
    return f"Please confirm your leave request:\n\n• Dates: {slots['start_date']} to {slots['end_date']} ({num_days} days)\n• Type: {slots['type']}\n• Reason: {slots['reason']}\n\nIs this correct? (yes/no)"

# =========================================================
# Other Agent Nodes (Simplified for demo)
# =========================================================

def ai_rag_agent_node(state: AgentState) -> AgentState:
    """RAG agent for general questions"""
    state["bot_response"] = "I'm here to help with workplace questions. For detailed information, please check our employee handbook or contact HR."
    state["next_node"] = "end"
    state["should_continue_session"] = False
    return state

def greeting_response_node(state: AgentState) -> AgentState:
    """Response to greetings"""
    state["bot_response"] = "Hello! I'm your workplace assistant. I can help with leave requests or answer general questions. How can I help you today?"
    state["next_node"] = "end"
    state["should_continue_session"] = False
    return state

# =========================================================
# Enhanced Graph with Session Routing
# =========================================================

def ai_router_condition(state: AgentState) -> Literal["rag_agent", "leave_agent", "greeting_response", "end"]:
    """Router condition with session awareness"""
    if state.get("should_continue_session", False):
        return "leave_agent"
    
    next_node = state.get("next_node", "rag_agent")
    return next_node

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("ai_intent_classifier", ai_intent_classifier_node)
workflow.add_node("rag_agent", ai_rag_agent_node)
workflow.add_node("leave_agent", ai_leave_agent_node)
workflow.add_node("greeting_response", greeting_response_node)

workflow.set_entry_point("ai_intent_classifier")
workflow.add_conditional_edges(
    "ai_intent_classifier",
    ai_router_condition,
    {
        "rag_agent": "rag_agent",
        "leave_agent": "leave_agent",
        "greeting_response": "greeting_response",
        "end": END
    }
)

workflow.add_edge("rag_agent", END)
workflow.add_edge("leave_agent", END)
workflow.add_edge("greeting_response", END)

app = workflow.compile()

# =========================================================
# Enhanced Processing with Session Management
# =========================================================

def process_message(user_id: str, user_message: str, conversation_history: List[Dict] = None) -> Dict:
    """Process message with session management"""
    if conversation_history is None:
        conversation_history = []
    
    initial_state: AgentState = {
        "user_id": user_id,
        "user_message": user_message,
        "bot_response": "",
        "intent": None,
        "next_node": None,
        "structured_data": {},
        "conversation_history": conversation_history,
        "session_id": None,
        "current_stage": None,
        "should_continue_session": False
    }
    
    # Execute the graph
    result = app.invoke(initial_state)
    
    # Update conversation history
    conversation_history.append({"user": user_message, "bot": result["bot_response"]})
    
    return {
        "response": result["bot_response"],
        "intent": result["intent"],
        "next_node": result["next_node"],
        "session_id": result.get("session_id"),
        "current_stage": result.get("current_stage"),
        "should_continue_session": result.get("should_continue_session", False)
    }

# =========================================================
# Flask API for JavaScript integration
# =========================================================
from flask import Flask, request, jsonify
import threading

flask_app = Flask(__name__)

@flask_app.route('/api/process-message', methods=['POST'])
def process_message_endpoint():
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')
    
    if not user_id or not message:
        return jsonify({'error': 'user_id and message are required'}), 400
    
    try:
        result = process_message(user_id, message)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask_app():
    flask_app.run(port=5000, debug=False, use_reloader=False)

# Start Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app, daemon=True)
flask_thread.start()

print("🤖 AI Workplace Assistant - Integrated with Google Chat")
print("🚀 Flask API running on port 5000")
print("=" * 60)