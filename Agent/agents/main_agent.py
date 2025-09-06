import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


class MainAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        # Store conversation state for each user
        self.user_states: Dict[str, Dict[str, Any]] = {}
        
        # Leave application fields
        self.required_fields = ["startDate", "endDate", "leaveType", "reason"]
    
    def get_or_create_user_state(self, user_email: str) -> Dict[str, Any]:
        """Get or create user state"""
        if user_email not in self.user_states:
            self.user_states[user_email] = {
                "currentIntent": "general",
                "leaveData": {},
                "conversationHistory": [],
                "isCollectingLeaveData": False,
                "lastMessage": None,
            }
        return self.user_states[user_email]
    
    async def classify_intent(self, user_message: str, user_email: str) -> str:
        """Classify user intent"""
        system_prompt = """You are an intent classification system for a workplace chatbot. 
        Classify the user's message into one of these intents:
        
        1. "leave_application" - User wants to apply for leave, vacation, time off, sick leave, etc.
        2. "general" - General conversation, greetings, questions, etc.
        
        Examples of leave_application:
        - "I want to take leave"
        - "Can I apply for vacation?"
        - "I need sick leave"
        - "I want time off next week"
        - "How do I request leave?"
        
        Respond with ONLY the intent name: "leave_application" or "general" """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content.lower().strip()
        except Exception as error:
            print(f"Error classifying intent: {error}")
            return "general"
    
    def is_leave_data_complete(self, leave_data: Dict[str, Any]) -> bool:
        """Check if all required leave data is collected"""
        return all(
            leave_data.get(field) and str(leave_data[field]).strip() != ""
            for field in self.required_fields
        )
    
    async def extract_leave_info(self, user_message: str, current_leave_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract leave information from user message"""
        system_prompt = f"""You are a data extraction system for leave applications. 
        Extract leave-related information from the user's message and return it in JSON format.
        
        Available fields to extract:
        - startDate: When the leave starts (format: YYYY-MM-DD or relative like "next Monday")
        - endDate: When the leave ends (format: YYYY-MM-DD or relative like "Friday")
        - leaveType: Type of leave (casual, sick, vacation, personal, emergency, etc.)
        - reason: Reason for the leave
        
        Current data already collected: {json.dumps(current_leave_data)}
        
        Only extract NEW information from the current message. If a field is already collected, don't overwrite it unless the user is correcting it.
        
        Return ONLY a JSON object with the extracted fields. If no new information is found, return an empty object {{}}.
        
        Examples:
        - "I want to take vacation from March 15 to March 20" → {{"startDate": "March 15", "endDate": "March 20", "leaveType": "vacation"}}
        - "I'm sick and need to take leave tomorrow" → {{"startDate": "tomorrow", "leaveType": "sick"}}
        - "I need personal leave next week" → {{"startDate": "next week", "leaveType": "personal"}}"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
            response = await self.llm.ainvoke(messages)
            extracted_data = json.loads(response.content)
            return extracted_data
        except Exception as error:
            print(f"Error extracting leave info: {error}")
            return {}
    
    async def generate_response(self, user_message: str, user_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversational response"""
        leave_data = user_state["leaveData"]
        is_collecting_leave_data = user_state["isCollectingLeaveData"]
        
        if is_collecting_leave_data:
            # Extract new information
            extracted_data = await self.extract_leave_info(user_message, leave_data)
            
            # Update leave data
            leave_data.update(extracted_data)
            
            # Check if all data is complete
            if self.is_leave_data_complete(leave_data):
                user_state["isCollectingLeaveData"] = False
                return {
                    "response": f"""Perfect! I have all the information needed for your leave application:
                    
📅 **Leave Details:**
- **Start Date:** {leave_data['startDate']}
- **End Date:** {leave_data['endDate']}
- **Leave Type:** {leave_data['leaveType']}
- **Reason:** {leave_data['reason']}

I'll now process your leave application. Please wait a moment...""",
                    "shouldInvokeLeaveAgent": True,
                    "leaveData": leave_data.copy(),
                }
            else:
                # Ask for missing information
                missing_fields = [
                    field for field in self.required_fields
                    if not leave_data.get(field) or str(leave_data[field]).strip() == ""
                ]
                
                field_prompts = {
                    "startDate": "When would you like your leave to start?",
                    "endDate": "When should your leave end?",
                    "leaveType": "What type of leave is this? (casual, sick, vacation, personal, etc.)",
                    "reason": "What's the reason for your leave?",
                }
                
                next_field = missing_fields[0]
                return {
                    "response": field_prompts[next_field],
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": leave_data.copy(),
                }
        else:
            # General conversation
            system_prompt = """You are a helpful workplace assistant. Be friendly, professional, and concise. 
            If the user mentions anything related to leave, vacation, time off, or sick days, gently guide them to provide more details about their leave request."""
            
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
                response = await self.llm.ainvoke(messages)
                
                return {
                    "response": response.content,
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": None,
                }
            except Exception as error:
                print(f"Error generating response: {error}")
                return {
                    "response": "I'm here to help! How can I assist you today?",
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": None,
                }
    
    async def process_message(self, user_message: str, user_email: str) -> Dict[str, Any]:
        """Main processing method"""
        user_state = self.get_or_create_user_state(user_email)
        
        # Classify intent
        intent = await self.classify_intent(user_message, user_email)
        
        # Update user state based on intent
        if intent == "leave_application" and not user_state["isCollectingLeaveData"]:
            user_state["isCollectingLeaveData"] = True
            user_state["leaveData"] = {}
            
            # Extract initial leave information
            extracted_data = await self.extract_leave_info(user_message, {})
            user_state["leaveData"].update(extracted_data)
            
            # If we have some data, continue collection
            if user_state["leaveData"]:
                missing_fields = [
                    field for field in self.required_fields
                    if not user_state["leaveData"].get(field) or str(user_state["leaveData"][field]).strip() == ""
                ]
                
                if missing_fields:
                    field_prompts = {
                        "startDate": "When would you like your leave to start?",
                        "endDate": "When should your leave end?",
                        "leaveType": "What type of leave is this? (casual, sick, vacation, personal, etc.)",
                        "reason": "What's the reason for your leave?",
                    }
                    
                    next_field = missing_fields[0]
                    return {
                        "response": field_prompts[next_field],
                        "shouldInvokeLeaveAgent": False,
                        "leaveData": user_state["leaveData"].copy(),
                    }
            else:
                return {
                    "response": "I'd be happy to help you apply for leave! Could you tell me when you'd like to start your leave?",
                    "shouldInvokeLeaveAgent": False,
                    "leaveData": {},
                }
        
        # Generate response
        result = await self.generate_response(user_message, user_state)
        
        # Update conversation history
        user_state["conversationHistory"].append({
            "user": user_message,
            "assistant": result["response"],
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last 10 messages to prevent memory issues
        if len(user_state["conversationHistory"]) > 10:
            user_state["conversationHistory"] = user_state["conversationHistory"][-10:]
        
        return result
    
    def reset_user_state(self, user_email: str) -> None:
        """Reset user state (useful for testing or when user wants to start over)"""
        if user_email in self.user_states:
            del self.user_states[user_email]