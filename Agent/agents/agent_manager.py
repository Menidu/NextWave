from typing import Dict, Any, Optional
from .main_agent import MainAgent
from .leave_agent import LeaveAgent


class AgentManager:
    def __init__(self):
        self.main_agent = MainAgent()
        self.leave_agent = LeaveAgent()
    
    async def process_message(self, user_message: str, user_email: str) -> Dict[str, Any]:
        """Main entry point for processing user messages"""
        try:
            print(f"🤖 Processing message from {user_email}: \"{user_message}\"")
            
            # Process with Main Agent first
            main_agent_result = await self.main_agent.process_message(user_message, user_email)
            
            # If Main Agent indicates we should invoke Leave Agent
            if (main_agent_result.get("shouldInvokeLeaveAgent") and 
                main_agent_result.get("leaveData")):
                print("📋 Leave data complete, invoking Leave Agent...")
                
                # Extracted leave data for debugging
                extracted_data = main_agent_result["leaveData"]
                print("Extracted leave data:", extracted_data)
                
                # Process with Leave Agent
                leave_agent_result = await self.leave_agent.process_leave_application(
                    extracted_data
                )
                
                # Combine results
                return {
                    "success": leave_agent_result["success"],
                    "message": leave_agent_result["message"],
                    "applicationId": leave_agent_result.get("applicationId"),
                    "status": leave_agent_result.get("status"),
                    "conflicts": leave_agent_result.get("conflicts", []),
                    "approvalStatus": leave_agent_result.get("approvalStatus"),
                    "error": leave_agent_result.get("error"),
                }
            else:
                # Return Main Agent response for general conversation or incomplete leave data
                return {
                    "success": True,
                    "message": main_agent_result["response"],
                    "isLeaveApplication": False,
                    "leaveData": main_agent_result.get("leaveData"),
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
        self.main_agent.reset_user_state(user_email)
    
    def get_all_applications(self) -> list:
        """Get all leave applications (for admin purposes)"""
        return self.leave_agent.get_all_applications()