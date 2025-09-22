import os
import json
import logging
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("RMAgent")


class RMAgent:
    """Resource Management Agent for user data and supervisor space retrieval."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.1,
        )
        
        # Mock database - in production, replace with actual database connection
        # These are realistic Google Chat space names (spaces/xxxxx format)
        self.user_database = {
            "john.doe@company.com": {
                "name": "John Doe",
                "email": "john.doe@company.com",
                "department": "Engineering",
                "supervisor_email": "jane.smith@company.com",
                "supervisor_name": "Jane Smith",
                "supervisor_chatspace": "spaces/wjkwk233",  # Supervisor's DM space
                "role": "Senior Developer",
                "employee_id": "EMP001",
                "hire_date": "2023-01-15",
                "location": "New York Office"
            },
            "jane.smith@company.com": {
                "name": "Jane Smith",
                "email": "jane.smith@company.com",
                "department": "Engineering",
                "supervisor_email": "bob.wilson@company.com",
                "supervisor_name": "Bob Wilson",
                "supervisor_chatspace": "spaces/abc123def",  # Supervisor's DM space
                "role": "Engineering Manager",
                "employee_id": "EMP002",
                "hire_date": "2022-06-01",
                "location": "New York Office"
            },
            "alice.johnson@company.com": {
                "name": "Alice Johnson",
                "email": "alice.johnson@company.com",
                "department": "HR",
                "supervisor_email": "carol.brown@company.com",
                "supervisor_name": "Carol Brown",
                "supervisor_chatspace": "spaces/xyz789ghi",  # Supervisor's DM space
                "role": "HR Specialist",
                "employee_id": "EMP003",
                "hire_date": "2023-03-10",
                "location": "Remote"
            }
        }
        
        logger.info("RMAgent initialized with mock database")
    
    async def get_user_data(self, user_email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data from database."""
        try:
            user_data = self.user_database.get(user_email.lower())
            if user_data:
                logger.info(f"Retrieved user data for: {user_email}")
                return user_data
            else:
                logger.warning(f"User not found in database: {user_email}")
                return None
        except Exception as error:
            logger.error(f"Error retrieving user data: {error}")
            return None
    
    async def get_supervisor_chatspace(self, user_email: str) -> Optional[str]:
        """Get supervisor's chatspace for a user.
        
        Returns the Google Chat space name (e.g., 'spaces/wjkwk233') where the supervisor
        receives DM messages. This is used to send approval request cards directly to
        the supervisor's DM space.
        """
        try:
            user_data = await self.get_user_data(user_email)
            if user_data and user_data.get("supervisor_chatspace"):
                chatspace = user_data["supervisor_chatspace"]
                logger.info(f"Retrieved supervisor chatspace for {user_email}: {chatspace}")
                return chatspace
            else:
                logger.warning(f"No supervisor chatspace found for: {user_email}")
                return None
        except Exception as error:
            logger.error(f"Error retrieving supervisor chatspace: {error}")
            return None
    
    async def get_supervisor_email(self, user_email: str) -> Optional[str]:
        """Get supervisor's email for a user."""
        try:
            user_data = await self.get_user_data(user_email)
            if user_data and user_data.get("supervisor_email"):
                logger.info(f"Retrieved supervisor email for {user_email}: {user_data['supervisor_email']}")
                return user_data["supervisor_email"]
            else:
                logger.warning(f"No supervisor email found for: {user_email}")
                return None
        except Exception as error:
            logger.error(f"Error retrieving supervisor email: {error}")
            return None
    
    async def process_rm_query(self, query: str, user_email: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Process RM-related queries using LLM with user context."""
        try:
            # Get user data if not provided
            if not user_context:
                user_context = await self.get_user_data(user_email)
            
            # Create context for LLM with fallback handling
            context_info = ""
            if user_context:
                context_info = f"""
USER CONTEXT:
- Name: {user_context.get('name', 'Unknown')}
- Email: {user_context.get('email', 'Unknown')}
- Department: {user_context.get('department', 'Unknown')}
- Role: {user_context.get('role', 'Unknown')}
- Employee ID: {user_context.get('employee_id', 'Unknown')}
- Hire Date: {user_context.get('hire_date', 'Unknown')}
- Location: {user_context.get('location', 'Unknown')}
- Supervisor: {user_context.get('supervisor_name', 'Unknown')} ({user_context.get('supervisor_email', 'Unknown')})
"""
            else:
                context_info = f"""
USER CONTEXT:
- Email: {user_email}
- Note: User information not found in database. Provide general assistance and suggest contacting HR for specific details.
"""
            
            system = f"""You are a Resource Management (RM) assistant. You help users with HR, employee, and organizational questions.

{context_info}

Instructions:
- Answer questions about employees, departments, organizational structure, HR policies, etc.
- Use the user context above to provide personalized responses when available
- If user data is not available, provide general assistance and suggest contacting HR
- If you don't have specific information, be honest and suggest contacting HR
- Be helpful, professional, and concise
- Respond in plain text only (no markdown, no bullets)
- Always maintain a friendly, helpful tone even when data is limited
"""
            
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=f"User query: {query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            logger.info(f"Processed RM query for {user_email} (context: {'available' if user_context else 'fallback'})")
            return response.content.strip()
            
        except Exception as error:
            logger.error(f"Error processing RM query: {error}")
            return "I'm sorry, I encountered an error while processing your RM query. Please try again or contact HR for assistance."
    
    async def is_rm_related_query(self, query: str) -> bool:
        """Determine if a query is RM-related using LLM."""
        try:
            system = """You are a query classifier. Determine if a user query is related to Resource Management (RM), HR, employees, organizational structure, or workplace administration.

RM-related topics include:
- Employee information and records
- Department and organizational structure
- HR policies and procedures
- Employee benefits and compensation
- Workplace administration
- Team and supervisor information
- Employee directory and contacts

Respond with only "YES" or "NO"."""
            
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=f"Query: {query}")
            ]
            
            response = await self.llm.ainvoke(messages)
            result = response.content.strip().upper()
            return result == "YES"
            
        except Exception as error:
            logger.error(f"Error classifying RM query: {error}")
            return False
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users from database (for admin purposes)."""
        return list(self.user_database.values())
    
    def add_user(self, user_data: Dict[str, Any]) -> bool:
        """Add a new user to the database."""
        try:
            email = user_data.get("email", "").lower()
            if email:
                self.user_database[email] = user_data
                logger.info(f"Added user to database: {email}")
                return True
            return False
        except Exception as error:
            logger.error(f"Error adding user: {error}")
            return False
    
    def get_chatspace_flow_explanation(self) -> str:
        """Explain how the chatspace flow works for documentation."""
        return """
CHATSPACE FLOW EXPLANATION:

1. User sends message → Google Chat webhook captures space name (e.g., "spaces/user123")
2. System processes message and returns response to same space
3. For leave approvals:
   - System looks up user's supervisor in database
   - Retrieves supervisor's DM chatspace (e.g., "spaces/wjkwk233")
   - Sends approval card directly to supervisor's DM space
4. Supervisor clicks Approve/Deny → Webhook processes decision
5. System notifies original requester in their space

This ensures messages go to the right people in the right spaces.
"""
