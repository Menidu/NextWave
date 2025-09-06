import express from "express";
import { google } from "googleapis";
import dotenv from "dotenv";
import cors from "cors";
import fs from "fs";
import axios from "axios";
import { AgentManager } from "./agents/agentManager.js";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 3005;
const SERVICE_ACCOUNT_FILE =
  process.env.SERVICE_ACCOUNT_KEY_FILE || "./service-account-key.json";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY; // <-- You set this in .env

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

// Initialize the multi-agent system
const agentManager = new AgentManager();

async function getAuthClient() {
  return await auth.getClient();
}

// ====== Gemini Flash 1.5 Integration ======

async function getGeminiReply(userMessage) {
  if (!GEMINI_API_KEY) {
    throw new Error("Gemini API key not set in environment variable.");
  }

  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;

  try {
    const response = await axios.post(endpoint, {
      contents: [{ role: "user", parts: [{ text: userMessage }] }],
    });

    const text = response.data.candidates?.[0]?.content?.parts?.[0]?.text;
    return text || "Sorry, I couldn’t understand that.";
  } catch (err) {
    console.error("❌ Error fetching Gemini response:", err.message);
    return "Something went wrong when contacting the AI model.";
  }
}

// ====== Shared Utility (Unused here, kept for reference) ======
async function sendMessageToUser(userEmail, messageText) {
  const authClient = await getAuthClient();

  const spaces = await chat.spaces.list({ auth: authClient, pageSize: 100 });
  const dmSpace = (spaces.data.spaces || []).find(
    (s) =>
      s.spaceType === "DIRECT_MESSAGE" &&
      s.singleUserBotDm?.user?.email === userEmail
  );

  if (!dmSpace) {
    throw new Error(
      `No DM space found with ${userEmail}. Ask them to message the bot first.`
    );
  }

  const response = await chat.spaces.messages.create({
    auth: authClient,
    parent: dmSpace.name,
    requestBody: { text: messageText },
  });

  return { result: response.data, space: dmSpace.name };
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

// Send to user
app.post("/chat/send-to-user", async (req, res) => {
  const { userEmail, message } = req.body;

  if (!userEmail || !message) {
    return res
      .status(400)
      .json({ success: false, error: "userEmail and message required" });
  }

  try {
    const result = await sendMessageToUser(userEmail, message);
    res.json({ success: true, ...result });
  } catch (error) {
    console.error("❌ Error sending to user:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get leave application status
app.get("/leave/status/:applicationId", async (req, res) => {
  const { applicationId } = req.params;

  try {
    const status = agentManager.getApplicationStatus(applicationId);
    if (!status) {
      return res.status(404).json({
        success: false,
        error: "Application not found",
      });
    }

    res.json({ success: true, application: status });
  } catch (error) {
    console.error("❌ Error getting application status:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Get all leave applications (admin endpoint)
app.get("/leave/applications", async (req, res) => {
  try {
    const applications = agentManager.getAllApplications();
    res.json({ success: true, applications });
  } catch (error) {
    console.error("❌ Error getting applications:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Reset user state (for testing)
app.post("/chat/reset-user", async (req, res) => {
  const { userEmail } = req.body;

  if (!userEmail) {
    return res
      .status(400)
      .json({ success: false, error: "userEmail required" });
  }

  try {
    agentManager.resetUserState(userEmail);
    res.json({ success: true, message: "User state reset successfully" });
  } catch (error) {
    console.error("❌ Error resetting user state:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Webhook handler
app.post("/chat/webhook", async (req, res) => {
  const event = req.body;

  console.log("🔔 Incoming webhook event:", JSON.stringify(event, null, 2));
  res.status(200).json({}); // Always respond immediately

  const spaceName = event?.chat?.messagePayload?.space?.name;
  const messageText = event?.chat?.messagePayload?.message?.text;
  const senderEmail = event?.chat?.messagePayload?.message?.sender?.email;

  if (!spaceName || !messageText) {
    console.warn("⚠️ No space name or message text found in event");
    return;
  }

  // Skip processing if the message is from the bot itself
  if (event?.chat?.messagePayload?.message?.sender?.type === "BOT") {
    console.log("🤖 Skipping bot message");
    return;
  }

  try {
    console.log(
      `💡 Processing message with multi-agent system: "${messageText}"`
    );

    // Process message with the multi-agent system
    const agentResult = await agentManager.processMessage(
      messageText,
      senderEmail
    );

    if (agentResult.success) {
      console.log(`💬 Sending agent response to ${spaceName}...`);
      await axios.post(`http://localhost:${PORT}/chat/send`, {
        spaceName,
        message: agentResult.message,
      });

      console.log(`✅ Replied to ${spaceName} with agent response.`);

      // Log additional details for leave applications
      if (agentResult.applicationId) {
        console.log(
          `📋 Leave application processed - ID: ${agentResult.applicationId}, Status: ${agentResult.status}`
        );
      }
    } else {
      console.error("❌ Agent processing failed:", agentResult.error);

      // Send error message to user
      await axios.post(`http://localhost:${PORT}/chat/send`, {
        spaceName,
        message:
          agentResult.message ||
          "I'm sorry, I encountered an error. Please try again.",
      });
    }
  } catch (error) {
    console.error("❌ Failed to process webhook:", error.message);

    // Send fallback error message
    try {
      await axios.post(`http://localhost:${PORT}/chat/send`, {
        spaceName,
        message:
          "I'm sorry, I'm having trouble processing your request right now. Please try again later.",
      });
    } catch (sendError) {
      console.error("❌ Failed to send error message:", sendError.message);
    }
  }
});

// ====== Start server ======
app.listen(PORT, () => {
  console.log(
    `🚀 Multi-Agent Workplace Bot running at http://localhost:${PORT}`
  );
  console.log("🤖 Agents initialized:");
  console.log("   - Main Agent (Conversational Router)");
  console.log("   - Leave Agent (LangGraph Workflow)");
  console.log("");
  console.log("📡 Endpoints:");
  console.log("   GET  /chat/spaces           → List spaces");
  console.log("   POST /chat/send             → Send to a space");
  console.log(
    "   POST /chat/send-to-user     → Send to a user (DM must exist)"
  );
  console.log("   POST /chat/webhook          → Handle incoming messages");
  console.log("   GET  /leave/status/:id      → Get leave application status");
  console.log("   GET  /leave/applications    → Get all applications (admin)");
  console.log("   POST /chat/reset-user       → Reset user state (testing)");
  console.log("");
  console.log("💬 The bot is ready to handle:");
  console.log("   - General workplace conversations");
  console.log("   - Leave application processing");
  console.log("   - Calendar conflict checking");
  console.log("   - Approval workflow simulation");
});
