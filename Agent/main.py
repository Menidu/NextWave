import os
import json
import httpx
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv

from agents.agent_manager import AgentManager

# Load environment variables
load_dotenv()

app = FastAPI(title="Multi-Agent Workplace Bot", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PORT = int(os.getenv("PORT", 3005))
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_KEY_FILE", "./service-account-key.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SCOPES = [
    "https://www.googleapis.com/auth/chat.bot",
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/chat.spaces",
]

# Global variables
auth_client = None
chat_service = None
agent_manager = None

# Pydantic models
class SendMessageRequest(BaseModel):
    spaceName: str
    message: str

class SendToUserRequest(BaseModel):
    userEmail: str
    message: str

class ResetUserRequest(BaseModel):
    userEmail: str

class WebhookEvent(BaseModel):
    chat: Optional[Dict[str, Any]] = None

# Initialize Google Auth and Chat service
def initialize_google_services():
    global auth_client, chat_service
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            credentials = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE, scopes=SCOPES
            )
            auth_client = credentials
            chat_service = build('chat', 'v1', credentials=credentials)
            print("✅ Google services initialized successfully")
        else:
            print("❌ Service account file not found")
    except Exception as error:
        print(f"❌ Error loading service account: {error}")
        raise

async def get_gemini_reply(user_message: str) -> str:
    """Get reply from Gemini Flash 1.5"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not set in environment variable.")
    
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json={
                "contents": [{"role": "user", "parts": [{"text": user_message}]}]
            })
            response.raise_for_status()
            
            data = response.json()
            text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
            return text or "Sorry, I couldn't understand that."
    except Exception as err:
        print(f"❌ Error fetching Gemini response: {err}")
        return "Something went wrong when contacting the AI model."

async def send_message_to_user(user_email: str, message_text: str) -> Dict[str, Any]:
    """Send message to user via Google Chat"""
    if not chat_service:
        raise HTTPException(status_code=500, detail="Google Chat service not initialized")
    
    try:
        # List spaces to find DM with user
        spaces_result = chat_service.spaces().list(pageSize=100).execute()
        spaces = spaces_result.get('spaces', [])
        
        # Find DM space with the user
        dm_space = None
        for space in spaces:
            if (space.get('spaceType') == 'DIRECT_MESSAGE' and 
                space.get('singleUserBotDm', {}).get('user', {}).get('email') == user_email):
                dm_space = space
                break
        
        if not dm_space:
            raise HTTPException(
                status_code=404, 
                detail=f"No DM space found with {user_email}. Ask them to message the bot first."
            )
        
        # Send message
        response = chat_service.spaces().messages().create(
            parent=dm_space['name'],
            body={'text': message_text}
        ).execute()
        
        return {
            "result": response,
            "space": dm_space['name']
        }
    except Exception as error:
        print(f"❌ Error sending message to user: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Startup event
@app.on_event("startup")
async def startup_event():
    global agent_manager
    initialize_google_services()
    agent_manager = AgentManager()
    print("🚀 Multi-Agent Workplace Bot started successfully")
    print("🤖 Agents initialized:")
    print("   - Main Agent (Conversational Router)")
    print("   - Leave Agent (LangGraph Workflow)")

# Health check
@app.get("/")
async def health_check():
    return {"success": True, "message": "Bot is running"}

# List spaces
@app.get("/chat/spaces")
async def list_spaces():
    if not chat_service:
        raise HTTPException(status_code=500, detail="Google Chat service not initialized")
    
    try:
        spaces_result = chat_service.spaces().list(pageSize=100).execute()
        return {
            "success": True,
            "spaces": spaces_result.get('spaces', [])
        }
    except Exception as error:
        print(f"❌ Error listing spaces: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Send message to space
@app.post("/chat/send")
async def send_message(request: SendMessageRequest):
    if not chat_service:
        raise HTTPException(status_code=500, detail="Google Chat service not initialized")
    
    try:
        response = chat_service.spaces().messages().create(
            parent=request.spaceName,
            body={'text': request.message}
        ).execute()
        
        return {"success": True, "result": response}
    except Exception as error:
        print(f"❌ Error sending message to space: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Send to user
@app.post("/chat/send-to-user")
async def send_to_user(request: SendToUserRequest):
    try:
        result = await send_message_to_user(request.userEmail, request.message)
        return {"success": True, **result}
    except Exception as error:
        print(f"❌ Error sending to user: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Get leave application status
@app.get("/leave/status/{application_id}")
async def get_application_status(application_id: str):
    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    try:
        status = agent_manager.get_application_status(application_id)
        if not status:
            raise HTTPException(status_code=404, detail="Application not found")
        
        return {"success": True, "application": status}
    except HTTPException:
        raise
    except Exception as error:
        print(f"❌ Error getting application status: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Get all leave applications (admin endpoint)
@app.get("/leave/applications")
async def get_all_applications():
    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    try:
        applications = agent_manager.get_all_applications()
        return {"success": True, "applications": applications}
    except Exception as error:
        print(f"❌ Error getting applications: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Reset user state (for testing)
@app.post("/chat/reset-user")
async def reset_user_state(request: ResetUserRequest):
    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")
    
    try:
        agent_manager.reset_user_state(request.userEmail)
        return {"success": True, "message": "User state reset successfully"}
    except Exception as error:
        print(f"❌ Error resetting user state: {error}")
        raise HTTPException(status_code=500, detail=str(error))

# Webhook handler
@app.post("/chat/webhook")
async def webhook_handler(request: Request):
    try:
        event = await request.json()
        print("🔔 Incoming webhook event:", json.dumps(event, indent=2))
        
        # Always respond immediately
        response = JSONResponse(content={}, status_code=200)
        
        # Extract event data
        space_name = event.get("chat", {}).get("messagePayload", {}).get("space", {}).get("name")
        message_text = event.get("chat", {}).get("messagePayload", {}).get("message", {}).get("text")
        sender_email = event.get("chat", {}).get("messagePayload", {}).get("message", {}).get("sender", {}).get("email")
        sender_type = event.get("chat", {}).get("messagePayload", {}).get("message", {}).get("sender", {}).get("type")
        
        if not space_name or not message_text:
            print("⚠️ No space name or message text found in event")
            return response
        
        # Skip processing if the message is from the bot itself
        if sender_type == "BOT":
            print("🤖 Skipping bot message")
            return response
        
        # Process in background (simulate async processing)
        import asyncio
        asyncio.create_task(process_webhook_message(space_name, message_text, sender_email))
        
        return response
        
    except Exception as error:
        print(f"❌ Failed to process webhook: {error}")
        return JSONResponse(content={}, status_code=200)

async def process_webhook_message(space_name: str, message_text: str, sender_email: str):
    """Process webhook message in background"""
    try:
        print(f"💡 Processing message with multi-agent system: \"{message_text}\"")
        
        if not agent_manager:
            print("❌ Agent manager not initialized")
            return
        
        # Process message with the multi-agent system
        agent_result = await agent_manager.process_message(message_text, sender_email)
        
        if agent_result["success"]:
            print(f"💬 Sending agent response to {space_name}...")
            
            # Send response using internal API
            async with httpx.AsyncClient() as client:
                await client.post(f"http://localhost:{PORT}/chat/send", json={
                    "spaceName": space_name,
                    "message": agent_result["message"]
                })
            
            print(f"✅ Replied to {space_name} with agent response.")
            
            # Log additional details for leave applications
            if agent_result.get("applicationId"):
                print(f"📋 Leave application processed - ID: {agent_result['applicationId']}, Status: {agent_result.get('status')}")
        else:
            print(f"❌ Agent processing failed: {agent_result.get('error')}")
            
            # Send error message to user
            error_message = agent_result.get("message") or "I'm sorry, I encountered an error. Please try again."
            async with httpx.AsyncClient() as client:
                await client.post(f"http://localhost:{PORT}/chat/send", json={
                    "spaceName": space_name,
                    "message": error_message
                })
    
    except Exception as error:
        print(f"❌ Failed to process webhook message: {error}")
        
        # Send fallback error message
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"http://localhost:{PORT}/chat/send", json={
                    "spaceName": space_name,
                    "message": "I'm sorry, I'm having trouble processing your request right now. Please try again later."
                })
        except Exception as send_error:
            print(f"❌ Failed to send error message: {send_error}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Multi-Agent Workplace Bot...")
    print(f"📡 Server will run on http://localhost:{PORT}")
    print("")
    print("📡 Endpoints:")
    print("   GET  /chat/spaces           → List spaces")
    print("   POST /chat/send             → Send to a space")
    print("   POST /chat/send-to-user     → Send to a user (DM must exist)")
    print("   POST /chat/webhook          → Handle incoming messages")
    print("   GET  /leave/status/:id      → Get leave application status")
    print("   GET  /leave/applications    → Get all applications (admin)")
    print("   POST /chat/reset-user       → Reset user state (testing)")
    print("")
    print("💬 The bot is ready to handle:")
    print("   - General workplace conversations")
    print("   - Leave application processing")
    print("   - Calendar conflict checking")
    print("   - Approval workflow simulation")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=PORT)