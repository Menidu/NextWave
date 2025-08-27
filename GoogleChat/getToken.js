import { GoogleAuth } from "google-auth-library";

async function main() {
  const auth = new GoogleAuth({
    keyFile: "chatbotproject-469508-5c96db8c80ac.json", // <-- path to your downloaded JSON
    scopes: ["https://www.googleapis.com/auth/chat.bot"],
  });

  const client = await auth.getClient();
  const token = await client.getAccessToken();

  console.log("Access Token:", token);
}

main().catch(console.error);
