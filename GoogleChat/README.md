# Multi-Agent Workplace Chatbot

A conversational workplace chatbot built with LangGraph that handles general conversations and leave applications through a multi-agent system.

## Architecture

### Main Agent (Conversational Router)

- Handles general workplace conversations
- Uses intent classification to detect leave application requests
- Collects leave information through natural conversation:
  - Start date
  - End date
  - Leave type (casual, sick, vacation, etc.)
  - Reason for leave
- Only invokes Leave Agent when all information is complete

### Leave Agent (LangGraph Workflow)

- Processes complete leave applications
- Checks for calendar conflicts
- Manages approval workflows
- Returns final status to user

## Features

- **Natural Conversation Flow**: Users can apply for leave through natural conversation
- **Intent Classification**: Automatically detects when users want to apply for leave
- **Persistent Data Collection**: Remembers information across multiple messages
- **Calendar Conflict Detection**: Simulates checking for scheduling conflicts
- **Approval Workflow**: Automatically approves certain types of leave or routes to manual review
- **Google Chat Integration**: Works seamlessly with Google Chat webhooks

## Setup

1. Install dependencies:

```bash
npm install
```

2. Set up environment variables in `.env`:

```
GEMINI_API_KEY=your_gemini_api_key
SERVICE_ACCOUNT_KEY_FILE=./chatbotproject-469508-5c96db8c80ac.json
PORT=3005
```

3. Start the server:

```bash
npm start
```

## Usage Examples

### General Conversation

```
User: "Hello, how are you?"
Bot: "Hello! I'm doing well, thank you. How can I help you today?"
```

### Leave Application

```
User: "I want to take vacation next week"
Bot: "I'd be happy to help you apply for leave! Could you tell me when you'd like to start your leave?"

User: "Monday"
Bot: "When should your leave end?"

User: "Friday"
Bot: "What type of leave is this? (casual, sick, vacation, personal, etc.)"

User: "vacation"
Bot: "What's the reason for your leave?"

User: "Family trip"
Bot: "Perfect! I have all the information needed for your leave application:
📅 Leave Details:
- Start Date: Monday
- End Date: Friday
- Leave Type: vacation
- Reason: Family trip

I'll now process your leave application. Please wait a moment..."

Bot: "✅ Leave Application Approved!
📅 Leave Details:
- Start Date: Monday
- End Date: Friday
- Leave Type: vacation
- Duration: 5 day(s)
- Reason: Family trip

Your leave has been automatically approved. Enjoy your time off!"
```

## API Endpoints

- `GET /chat/spaces` - List available chat spaces
- `POST /chat/send` - Send message to a space
- `POST /chat/send-to-user` - Send message to a specific user
- `POST /chat/webhook` - Handle incoming Google Chat messages
- `GET /leave/status/:id` - Get leave application status
- `GET /leave/applications` - Get all applications (admin)
- `POST /chat/reset-user` - Reset user state (testing)

## Leave Types Supported

- **Casual Leave**: Auto-approved for 1-2 days
- **Sick Leave**: Auto-approved
- **Vacation Leave**: May require manual approval for long periods
- **Personal Leave**: Requires manual approval
- **Emergency Leave**: Requires manual approval

## Calendar Conflict Detection

The system simulates calendar conflict checking:

- Long vacations may conflict with team meetings
- Sick leave during month-end may require review
- Conflicts trigger manual review process

## Approval Workflow

1. **Auto-Approval**: Short casual leaves and sick leaves
2. **Manual Review**: Long vacations, personal leaves, and applications with conflicts
3. **Status Tracking**: All applications are tracked with unique IDs

## Development

The system is built with:

- **LangGraph**: For the Leave Agent workflow
- **LangChain**: For LLM integration
- **Google Generative AI**: For conversation and intent classification
- **Express.js**: For the web server
- **Google Chat API**: For chat integration

## Testing

Use the reset endpoint to clear user state during testing:

```bash
curl -X POST http://localhost:3005/chat/reset-user \
  -H "Content-Type: application/json" \
  -d '{"userEmail": "user@example.com"}'
```
