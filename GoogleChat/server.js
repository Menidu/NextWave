// Google Chat Bot with LangGraph Agent Integration
import express from "express";
import { google } from "googleapis";
import dotenv from "dotenv";
import cors from "cors";
import fs from "fs";
import axios from "axios";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 3005;
const SERVICE_ACCOUNT_FILE =
  process.env.SERVICE_ACCOUNT_KEY_FILE || "./service-account-key.json";

const SCOPES = [
  "https://www.googleapis.com/auth/chat.bot",
  "https://www.googleapis.com/auth/chat.messages",
  "https://www.googleapis.com/auth/chat.spaces",
];

// ====== Google Auth Setup ======
let auth;
try {
  const serviceAccountKey = JSON.parse(
    fs.readFileSync(SERVICE_ACCOUNT_FILE, "utf8")
  );
  auth = new google.auth.GoogleAuth({
    credentials: serviceAccountKey,
    scopes: SCOPES,
  });
} catch (error) {
  console.error("❌ Error loading service account:", error.message);
  process.exit(1);
}

const chat = google.chat({ version: "v1", auth });

async function getAuthClient() {
  return await auth.getClient();
}

// ====== LangGraph Agent Integration ======
// This will be our interface to the Python agent
async function getAgentResponse(
  spaceId,
  userMessage,
  conversationHistory = []
) {
  try {
    // In a production environment, you might want to run the Python agent
    // as a separate service and call it via HTTP or gRPC

    // For this demo, we'll simulate the agent response
    // In a real implementation, you would call your Python agent here

    // Simulate agent processing
    const response = await processMessageWithAgent(
      spaceId,
      userMessage,
      conversationHistory
    );
    return response;
  } catch (error) {
    console.error("❌ Error getting agent response:", error.message);
    return "I'm having trouble processing your request right now. Please try again later.";
  }
}

// Mock function to simulate agent processing
// In production, this would connect to your actual Python agent
async function processMessageWithAgent(
  spaceId,
  userMessage,
  conversationHistory
) {
  // This is a simplified simulation of the agent's behavior
  // In a real implementation, you would call your Python code here

  // Check if we have an active session for this space
  const hasActiveSession = checkActiveSession(spaceId);

  if (hasActiveSession) {
    // Continue leave process
    return continueLeaveProcess(spaceId, userMessage);
  } else if (
    userMessage.toLowerCase().includes("leave") ||
    userMessage.toLowerCase().includes("time off") ||
    userMessage.toLowerCase().includes("vacation")
  ) {
    // Start leave process
    return startLeaveProcess(spaceId, userMessage);
  } else {
    // General response
    return "I'm here to help with leave requests and workplace questions. How can I assist you today?";
  }
}

// Mock session management
const activeSessions = new Map();

function checkActiveSession(spaceId) {
  return activeSessions.has(spaceId);
}

function startLeaveProcess(spaceId, userMessage) {
  activeSessions.set(spaceId, {
    stage: "start",
    data: {},
  });

  return "I'd be happy to help you apply for leave. When would you like to take time off? Please provide start and end dates (e.g., 'next Monday to Friday').";
}

function continueLeaveProcess(spaceId, userMessage) {
  const session = activeSessions.get(spaceId);

  switch (session.stage) {
    case "start":
      // Try to extract dates
      const dates = extractDates(userMessage);
      if (dates.start && dates.end) {
        session.data.startDate = dates.start;
        session.data.endDate = dates.end;
        session.stage = "type";
        return "Great! What type of leave would you like to take? (annual, sick, personal, etc.)";
      } else {
        return "I need to know the dates for your leave. Please provide start and end dates (e.g., 'next Monday to Friday').";
      }

    case "type":
      const type = extractLeaveType(userMessage);
      if (type) {
        session.data.type = type;
        session.stage = "reason";
        return `${type} leave, noted! Could you please tell me the reason for your leave?`;
      } else {
        return "What type of leave would you like to take? (annual, sick, personal, emergency, vacation)";
      }

    case "reason":
      session.data.reason = userMessage;
      session.stage = "confirmation";
      return `Please confirm your leave request:\n\n• Dates: ${session.data.startDate} to ${session.data.endDate}\n• Type: ${session.data.type}\n• Reason: ${session.data.reason}\n\nIs this correct? (yes/no)`;

    case "confirmation":
      if (
        userMessage.toLowerCase().includes("yes") ||
        userMessage.toLowerCase().includes("correct") ||
        userMessage.toLowerCase().includes("confirm")
      ) {
        // Complete the process
        activeSessions.delete(spaceId);
        return "✅ Leave request created! I'll now check for calendar conflicts and send it for approval.";
      } else {
        activeSessions.delete(spaceId);
        return "Okay, let's start over. When would you like to take time off?";
      }

    default:
      activeSessions.delete(spaceId);
      return "I'm not sure what to do next. Let's start over.";
  }
}

// Helper functions for mock implementation
function extractDates(message) {
  // Simple date extraction - in real implementation, use proper NLP
  const datePattern = /(\d{1,2}\/\d{1,2}\/\d{4})|(\w+ \d{1,2})|(next \w+)/gi;
  const matches = message.match(datePattern);

  if (matches && matches.length >= 2) {
    return { start: matches[0], end: matches[1] };
  }

  return { start: null, end: null };
}

function extractLeaveType(message) {
  const types = ["annual", "sick", "personal", "emergency", "vacation"];
  for (const type of types) {
    if (message.toLowerCase().includes(type)) {
      return type;
    }
  }
  return null;
}

// ====== Routes ======

// Health check
app.get("/", (req, res) => {
  res.json({ success: true, message: "Bot is running" });
});

// List spaces
app.get("/chat/spaces", async (req, res) => {
  try {
    const authClient = await getAuthClient();
    const spaces = await chat.spaces.list({
      auth: authClient,
      pageSize: 100,
    });

    res.json({
      success: true,
      spaces: spaces.data.spaces || [],
    });
  } catch (error) {
    console.error("❌ Error listing spaces:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Send message to space
app.post("/chat/send", async (req, res) => {
  const { spaceName, message } = req.body;

  if (!spaceName || !message) {
    return res
      .status(400)
      .json({ success: false, error: "spaceName and message are required" });
  }

  try {
    const authClient = await getAuthClient();
    const response = await chat.spaces.messages.create({
      auth: authClient,
      parent: spaceName,
      requestBody: { text: message },
    });

    res.json({ success: true, result: response.data });
  } catch (error) {
    console.error("❌ Error sending message to space:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Webhook handler - Main integration point
app.post("/chat/webhook", async (req, res) => {
  const event = req.body;

  console.log("🔔 Incoming webhook event:", JSON.stringify(event, null, 2));

  // Always respond immediately to acknowledge receipt
  res.status(200).json({});

  const spaceName = event.space?.name;
  const messageText = event.message?.text;
  const senderEmail = event.message?.sender?.email;

  if (!spaceName || !messageText) {
    console.warn("⚠️ No space name or message text found in event");
    return;
  }

  // Ignore messages from the bot itself to prevent loops
  if (senderEmail && senderEmail.includes("gserviceaccount.com")) {
    console.log("🤖 Ignoring message from bot itself");
    return;
  }

  try {
    console.log(`💡 Getting agent response for message: "${messageText}"`);

    // Use spaceName as the user ID for session management
    const agentResponse = await getAgentResponse(spaceName, messageText);

    console.log(`💬 Sending agent reply to ${spaceName}...`);

    // Send the response back to the Google Chat space
    const authClient = await getAuthClient();
    await chat.spaces.messages.create({
      auth: authClient,
      parent: spaceName,
      requestBody: { text: agentResponse },
    });

    console.log(`✅ Replied to ${spaceName} with agent response.`);
  } catch (error) {
    console.error("❌ Failed to process webhook:", error.message);
  }
});

// ====== Start server ======
app.listen(PORT, () => {
  console.log(`🚀 Bot running at http://localhost:${PORT}`);
  console.log("Endpoints:");
  console.log("   GET  /chat/spaces        → List spaces");
  console.log("   POST /chat/send          → Send to a space");
  console.log("   POST /chat/webhook       → Handle incoming messages");
});
