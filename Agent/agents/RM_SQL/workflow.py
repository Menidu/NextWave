from langgraph.graph import StateGraph, END, START
from .state import RMState
from langchain.agents import AgentType
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Workflow:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Initialize database connection
        self.mysql_username = os.getenv("MYSQL_USERNAME")
        self.mysql_password = os.getenv("MYSQL_PASSWORD")
        self.mysql_host = os.getenv("MYSQL_HOST")
        self.mysql_port = os.getenv("MYSQL_PORT")
        self.database_name = os.getenv("MYSQL_DATABASE")
        self.mysql_uri = f'mysql+mysqlconnector://{self.mysql_username}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.database_name}'
        self.db = SQLDatabase.from_uri(self.mysql_uri)
        
        # Initialize agent and workflow
        self.agent = self.create_agent()
        self.workflow = self.build_workflow()

    def create_agent(self):
        """Create the agent"""
        return create_sql_agent(
            llm=self.llm, 
            db=self.db, 
            verbose=True, 
            handle_parsing_errors=True, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

    def agent_node(self, state: RMState):
        """Agent node that processes the query and returns only the final AI text response"""
        try:
            query = state.get("query", "")
            
            # Use the new SQL agent invocation pattern
            result = self.agent.invoke({"input": query})

            response_text = ""

            # Extract the final AI response from the result
            if isinstance(result, dict):
                # Check for 'output' key first (most common)
                if 'output' in result:
                    response_text = str(result['output']).strip()
                # Check for 'messages' key
                elif 'messages' in result:
                    messages = result['messages']
                    if messages:
                        # Find the last AI message with content
                        for message in reversed(messages):
                            if (hasattr(message, '__class__') and 
                                message.__class__.__name__ == "AIMessage" and 
                                hasattr(message, 'content') and 
                                message.content):
                                content = str(message.content).strip()
                                if content:
                                    response_text = content
                                    break
                # Check for direct string response
                elif 'response' in result:
                    response_text = str(result['response']).strip()
            
            # Handle case where result has messages attribute
            elif hasattr(result, 'messages') and result.messages:
                # Get all AI messages
                ai_messages = [msg for msg in result.messages if hasattr(msg, '__class__') and msg.__class__.__name__ == "AIMessage"]
                
                if ai_messages:
                    # Try to get the last AI message with meaningful content
                    for message in reversed(ai_messages):
                        if hasattr(message, 'content') and message.content:
                            content = str(message.content).strip()
                            # Skip empty content or just function calls
                            if content and not content.startswith('(function='):
                                response_text = content
                                break
                    
                    # If still no response, try any AI message with content
                    if not response_text:
                        for message in ai_messages:
                            if hasattr(message, 'content') and message.content:
                                content = str(message.content).strip()
                                if content:
                                    response_text = content
                                    break
            
            # Handle direct string result
            elif isinstance(result, str):
                response_text = result.strip()
            
            # Fallback
            if not response_text:
                response_text = "I couldn't process your request properly."

            # Update state with only the final AI response
            return {
                **state,
                "response": response_text,
                "error": None
            }

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "response": "",
                "error": error_msg
            }

    def build_workflow(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(RMState)
        
        # Add nodes
        workflow.add_node("agent", self.agent_node)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)
        
        return workflow.compile()

    def run(self, query: str):
        """Run the SQL agent with a query"""
        try:
            # Initialize state
            initial_state = {
                "query": query,
                "response": "",
                "error": None
            }

            # Run the workflow
            final_state = self.workflow.invoke(initial_state)

            # Extract the clean response
            response_text = final_state.get("response", "").strip()
            error_text = final_state.get("error", None)

            # Return only the essential information
            if error_text:
                return {
                    "query": query,
                    "response": "",
                    "error": error_text,
                    "success": False
                }
            else:
                return {
                    "query": query,
                    "response": response_text,
                    "success": True,
                    "error": None
                }

        except Exception as e:
            error_msg = f"SQL Workflow error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "query": query,
                "response": "",
                "error": error_msg,
                "success": False
            }