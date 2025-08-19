// Google Chat Bot (Push-only, deployable on Vercel)
// Install: npm install express googleapis dotenv cors

const express = require("express");
const { google } = require("googleapis");
require("dotenv").config();
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

// ====== Service Account Config ======
// You can store your service account JSON as an ENV variable for Vercel:
// SERVICE_ACCOUNT_KEY_JSON={"type":"...","project_id":"..."}
let serviceAccountKey;
try {
  if (process.env.SERVICE_ACCOUNT_KEY_JSON) {
    serviceAccountKey = JSON.parse(process.env.SERVICE_ACCOUNT_KEY_JSON);
  } else {
    // fallback to local file
    const fs = require("fs");
    serviceAccountKey = JSON.parse(
      fs.readFileSync("./service-account-key.json", "utf8")
    );
  }
} catch (error) {
  console.error("Error loading service account:", error.message);
  process.exit(1);
}

const SCOPES = [
  "https://www.googleapis.com/auth/chat.bot",
  "https://www.googleapis.com/auth/chat.messages",
  "https://www.googleapis.com/auth/chat.spaces",
];

const auth = new google.auth.GoogleAuth({
  credentials: serviceAccountKey,
  scopes: SCOPES,
});

// Init Chat API
const chat = google.chat({ version: "v1", auth });

// Helper: get auth client
async function getAuthClient() {
  return await auth.getClient();
}

// ========= ROUTES ===========

// Health check
app.get("/", (req, res) => {
  res.json({ success: true, message: "Bot is running" });
});

// 1. List all spaces the bot is in
app.get("/chat/spaces", async (req, res) => {
  try {
    const authClient = await getAuthClient();
    const spaces = await chat.spaces.list({ auth: authClient, pageSize: 100 });

    res.json({ success: true, spaces: spaces.data.spaces || [] });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// 2. Send a message to an existing space (DM or room)
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
    res.status(500).json({ success: false, error: error.message });
  }
});

// 3. Shortcut: Send to user (only if DM already exists)
app.post("/chat/send-to-user", async (req, res) => {
  const { userEmail, message } = req.body;
  if (!userEmail || !message) {
    return res
      .status(400)
      .json({ success: false, error: "userEmail and message required" });
  }

  try {
    const authClient = await getAuthClient();
    const spaces = await chat.spaces.list({ auth: authClient, pageSize: 100 });
    const dmSpace = (spaces.data.spaces || []).find(
      (s) =>
        s.spaceType === "DIRECT_MESSAGE" &&
        s.singleUserBotDm?.user?.email === userEmail
    );

    if (!dmSpace) {
      return res.status(400).json({
        success: false,
        error: `No DM space found with ${userEmail}. Ask them to message the bot first.`,
      });
    }

    const response = await chat.spaces.messages.create({
      auth: authClient,
      parent: dmSpace.name,
      requestBody: { text: message },
    });

    res.json({ success: true, result: response.data, space: dmSpace.name });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// ====== START SERVER ======
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Bot running at port ${PORT}`);
  console.log("Endpoints:");
  console.log("   GET  /chat/spaces        → List spaces");
  console.log("   POST /chat/send          → Send to a space");
  console.log(
    "   POST /chat/send-to-user  → Send to a user (requires DM exists)"
  );
});

s;
