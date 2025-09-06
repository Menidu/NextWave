# Configuration Guide

## Environment Variables

Create a `.env` file in the GoogleChat directory with the following variables:

```env
# Google Generative AI API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Google Service Account Key File Path
SERVICE_ACCOUNT_KEY_FILE=./chatbotproject-469508-5c96db8c80ac.json

# Server Port
PORT=3005
```

## Getting Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key and add it to your `.env` file

## Google Service Account Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Chat API
4. Create a service account:
   - Go to IAM & Admin > Service Accounts
   - Click "Create Service Account"
   - Give it a name and description
   - Grant it the necessary permissions for Google Chat
5. Create a key for the service account:
   - Click on the service account
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format
   - Download the key file
6. Place the key file in the GoogleChat directory
7. Update the `SERVICE_ACCOUNT_KEY_FILE` path in your `.env` file

## Testing the System

1. Install dependencies:

```bash
npm install
```

2. Start the server:

```bash
npm start
```

3. Test the agents (optional):

```bash
node test-agents.js
```

## Google Chat Bot Setup

1. In Google Cloud Console, go to Google Chat API
2. Create a new bot configuration
3. Set the webhook URL to: `https://your-domain.com/chat/webhook`
4. Configure the bot's display name and avatar
5. Deploy the bot to your Google Workspace

## Troubleshooting

### Common Issues

1. **"Gemini API key not set"**: Make sure your `GEMINI_API_KEY` is set in the `.env` file
2. **"Error loading service account"**: Check that the service account key file exists and is valid
3. **"No DM space found"**: The user needs to message the bot first to create a DM space
4. **Webhook not receiving messages**: Ensure your webhook URL is publicly accessible and the bot is properly configured

### Debug Mode

To enable more detailed logging, you can modify the console.log statements in the code or add a debug environment variable.

### Testing Locally

For local testing, you can use ngrok to expose your local server:

```bash
npm run tunnel
```

Then use the ngrok URL as your webhook URL in the Google Chat bot configuration.
