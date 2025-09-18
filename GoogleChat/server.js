import express from "express";
import { google } from "googleapis";
import dotenv from "dotenv";
import cors from "cors";
import fs from "fs";
import axios from "axios";
import { AgentManager } from "./agents/agentManager.js";
import {
  getUserProfile,
  setSupervisorForUser,
  setSupervisorSpace,
} from "./agents/userDirectory.js";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

const PORT = process.env.PORT || 3005;
const SUPERVISOR_EMAIL =
  process.env.SUPERVISOR_EMAIL || process.env.GOOGLE_CHAT_SUPERVISOR_EMAIL;
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

// ====== Shared Utility ======
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

// Send a cardsV2 interactive card to a user's DM (DM must already exist)
async function sendCardToUser(userEmail, cardV2) {
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
    requestBody: { cardsV2: [cardV2] },
  });

  return { result: response.data, space: dmSpace.name };
}

async function getSupervisorEmailFor(userEmail) {
  const profile = getUserProfile(userEmail);
  return (
    profile?.supervisorEmail ||
    process.env.SUPERVISOR_EMAIL ||
    process.env.GOOGLE_CHAT_SUPERVISOR_EMAIL ||
    null
  );
}

// Find supervisor DM space and cache it for the user; returns {spaceName}
async function getOrCacheSupervisorDmSpace(userEmail) {
  const supervisorEmail = await getSupervisorEmailFor(userEmail);
  if (!supervisorEmail) return { spaceName: null, supervisorEmail: null };

  const profile = getUserProfile(userEmail);
  if (profile.supervisorSpaceName) {
    return { spaceName: profile.supervisorSpaceName, supervisorEmail };
  }

  const authClient = await getAuthClient();
  const spaces = await chat.spaces.list({ auth: authClient, pageSize: 100 });
  const dmSpace = (spaces.data.spaces || []).find(
    (s) =>
      s.spaceType === "DIRECT_MESSAGE" &&
      s.singleUserBotDm?.user?.email === supervisorEmail
  );
  if (!dmSpace) return { spaceName: null, supervisorEmail };

  setSupervisorSpace(userEmail, dmSpace.name);
  return { spaceName: dmSpace.name, supervisorEmail };
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

// Set supervisor for a user
app.post("/users/:email/supervisor", async (req, res) => {
  const { email } = req.params;
  const { supervisorEmail } = req.body || {};
  if (!email || !supervisorEmail) {
    return res
      .status(400)
      .json({ success: false, error: "email and supervisorEmail required" });
  }
  try {
    setSupervisorForUser(email, supervisorEmail);
    res.json({ success: true, email, supervisorEmail });
  } catch (error) {
    console.error("❌ Error setting supervisor:", error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Webhook handler
app.post("/chat/webhook", async (req, res) => {
  const event = req.body;

  console.log("🔔 Incoming webhook event:", JSON.stringify(event, null, 2));
  res.status(200).json({}); // Always respond immediately

  // Handle interactive card actions first (approval/denial)
  try {
    const action = event?.action || event?.common?.invokedFunction || null;
    const actionMethodName =
      action?.actionMethodName || event?.actionMethodName;
    if (actionMethodName === "leave_approval") {
      const paramsArray = action?.parameters || event?.parameters || [];
      const params = Object.fromEntries(
        paramsArray.map((p) => [p.key, p.value])
      );

      const applicationId = params.applicationId;
      const decision = params.decision;
      const approverEmail =
        event?.chat?.messagePayload?.message?.sender?.email ||
        event?.user?.email ||
        null;

      if (!applicationId || !decision) {
        console.warn("⚠️ Missing applicationId/decision in action payload");
        return;
      }

      const result = agentManager.leaveAgent.setApprovalStatus(
        applicationId,
        decision === "approved" ? "approved" : "denied",
        approverEmail
      );

      if (!result.success) {
        console.error("❌ Failed to update approval status:", result.error);
        return;
      }

      const application = result.application;
      const requesterEmail = application.requesterEmail;

      if (requesterEmail) {
        const requesterMsg =
          decision === "approved"
            ? `✅ Your leave request ${applicationId} was approved.\n\nDetails\n- Start: ${application.startDate}\n- End: ${application.endDate}\n- Type: ${application.leaveType}`
            : `❌ Your leave request ${applicationId} was denied.`;
        try {
          await sendMessageToUser(requesterEmail, requesterMsg);
        } catch (e) {
          console.error("❌ Failed to notify requester:", e.message);
        }
      }

      if (approverEmail) {
        const approverMsg = `Recorded your decision (${decision}) for request ${applicationId}.`;
        try {
          await sendMessageToUser(approverEmail, approverMsg);
        } catch (e) {
          console.error("❌ Failed to notify approver:", e.message);
        }
      }

      return; // Action handled; stop further processing
    }
  } catch (actionErr) {
    console.error("❌ Error handling action in webhook:", actionErr.message);
  }

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

      // If pending approval, send approval card to requester's supervisor
      if (agentResult.approvalStatus === "pending_approval") {
        try {
          const { spaceName, supervisorEmail } =
            await getOrCacheSupervisorDmSpace(senderEmail);
          const targetSupervisorEmail = supervisorEmail || SUPERVISOR_EMAIL;
          if (!spaceName && !targetSupervisorEmail) {
            console.warn(
              "⚠️ No supervisor email or DM space configured for requester; cannot send approval card"
            );
            return;
          }
          const card = {
            cardId: `leave-approval-${agentResult.applicationId}`,
            card: {
              header: {
                title: "Leave approval request",
                subtitle: `Application ${agentResult.applicationId}`,
              },
              sections: [
                {
                  widgets: [
                    { textParagraph: { text: agentResult.message } },
                    {
                      buttonList: {
                        buttons: [
                          {
                            text: "Approve",
                            onClick: {
                              action: {
                                actionMethodName: "leave_approval",
                                parameters: [
                                  {
                                    key: "applicationId",
                                    value: agentResult.applicationId,
                                  },
                                  { key: "decision", value: "approved" },
                                ],
                              },
                            },
                          },
                          {
                            text: "Deny",
                            onClick: {
                              action: {
                                actionMethodName: "leave_approval",
                                parameters: [
                                  {
                                    key: "applicationId",
                                    value: agentResult.applicationId,
                                  },
                                  { key: "decision", value: "denied" },
                                ],
                              },
                            },
                          },
                        ],
                      },
                    },
                  ],
                },
              ],
            },
          };

          if (spaceName) {
            const authClient = await getAuthClient();
            await chat.spaces.messages.create({
              auth: authClient,
              parent: spaceName,
              requestBody: { cardsV2: [card] },
            });
            console.log(
              `📝 Sent approval card to cached supervisor DM ${spaceName} for ${agentResult.applicationId}`
            );
          } else {
            await sendCardToUser(targetSupervisorEmail, card);
            console.log(
              `📝 Sent approval card to supervisor ${targetSupervisorEmail} for ${agentResult.applicationId}`
            );
          }
        } catch (cardError) {
          console.error("❌ Failed to send approval card:", cardError.message);
        }
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

// Handle Google Chat interactive card actions for approvals
app.post("/chat/action", async (req, res) => {
  const event = req.body;
  res.status(200).json({});

  try {
    const action = event?.action || event?.common?.invokedFunction || null;
    const actionMethodName =
      action?.actionMethodName || event?.actionMethodName;
    if (actionMethodName !== "leave_approval") return;

    const paramsArray = action?.parameters || event?.parameters || [];
    const params = Object.fromEntries(paramsArray.map((p) => [p.key, p.value]));
    const applicationId = params.applicationId;
    const decision = params.decision;
    const approverEmail =
      event?.chat?.messagePayload?.message?.sender?.email ||
      event?.user?.email ||
      null;

    if (!applicationId || !decision) {
      console.warn("⚠️ Missing applicationId/decision in action payload");
      return;
    }

    const result = agentManager.leaveAgent.setApprovalStatus(
      applicationId,
      decision === "approved" ? "approved" : "denied",
      approverEmail
    );

    if (!result.success) {
      console.error("❌ Failed to update approval status:", result.error);
      return;
    }

    const application = result.application;
    const requesterEmail = application.requesterEmail;

    if (requesterEmail) {
      const requesterMsg =
        decision === "approved"
          ? `✅ Your leave request ${applicationId} was approved.\n\nDetails\n- Start: ${application.startDate}\n- End: ${application.endDate}\n- Type: ${application.leaveType}`
          : `❌ Your leave request ${applicationId} was denied.`;
      try {
        await sendMessageToUser(requesterEmail, requesterMsg);
      } catch (e) {
        console.error("❌ Failed to notify requester:", e.message);
      }
    }

    if (approverEmail) {
      const approverMsg = `Recorded your decision (${decision}) for request ${applicationId}.`;
      try {
        await sendMessageToUser(approverEmail, approverMsg);
      } catch (e) {
        console.error("❌ Failed to notify approver:", e.message);
      }
    }
  } catch (err) {
    console.error("❌ Error handling card action:", err.message);
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
