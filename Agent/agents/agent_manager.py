from typing import Dict, Any, Optional
from .agentic_agent import AgenticAgent
from .leave_agent import LeaveAgent


class AgentManager:
    def __init__(self):
        self.agentic = AgenticAgent()
        self.leave_agent = LeaveAgent()
    
    async def process_message(self, user_message: str, user_email: str, user_display_name: str = "", space_name: str = "") -> Dict[str, Any]:
        """Main entry point for processing user messages"""
        try:
            print(f"🤖 Processing message from {user_email}: \"{user_message}\"")
            
            # Process with the agentic graph
            agent_result = await self.agentic.process_message(user_message, user_email, user_display_name)

            route = agent_result.get("route")
            leave_data = agent_result.get("leaveData") or {}
            response_text = agent_result.get("response") or ""

            # If agent decided to run workflow or we have all fields, invoke Leave Agent
            required = ["startDate", "endDate", "leaveType", "reason", "supervisorEmail"]
            has_all = isinstance(leave_data, dict) and all(leave_data.get(f) for f in required)
            if route == "run_leave_workflow" or has_all:
                print("📋 Leave data ready, invoking Leave Agent...")
                extracted_data = dict(leave_data)
                extracted_data.setdefault("requesterEmail", user_email)
                if space_name:
                    extracted_data.setdefault("requesterSpaceName", space_name)

                leave_agent_result = await self.leave_agent.process_leave_application(extracted_data)

                return {
                    "success": leave_agent_result["success"],
                    "message": leave_agent_result["message"],
                    "applicationId": leave_agent_result.get("applicationId"),
                    "status": leave_agent_result.get("status"),
                    "conflicts": leave_agent_result.get("conflicts", []),
                    "approvalStatus": leave_agent_result.get("approvalStatus"),
                    "error": leave_agent_result.get("error"),
                }

            # Otherwise return the agentic assistant's response (collection or general chat)
            return {
                "success": True,
                "message": response_text,
                "isLeaveApplication": False,
                "leaveData": leave_data,
            }
        
        except Exception as error:
            print(f"❌ Error in AgentManager: {error}")
            return {
                "success": False,
                "message": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "error": str(error),
            }
    
    def get_application_status(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get application status"""
        return self.leave_agent.get_application_status(application_id)
    
    def reset_user_state(self, user_email: str) -> None:
        """Reset user state (useful for testing)"""
        # For AgenticAgent with SQLite checkpointing, simplest is to change thread_id externally.
        # Keep method for compatibility; no-op here.
        self.agentic.reset_thread(user_email)
    
    def get_all_applications(self) -> list:
        """Get all leave applications (for admin purposes)"""
        return self.leave_agent.get_all_applications()