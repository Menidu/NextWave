import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export class MainAgent {
  constructor() {
    this.llm = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      apiKey: process.env.GEMINI_API_KEY,
    });

    // Store conversation state for each user
    this.userStates = new Map();

    // Leave application fields
    this.requiredFields = ["startDate", "endDate", "leaveType", "reason"];
  }

  // Get or create user state
  getOrCreateUserState(userEmail) {
    if (!this.userStates.has(userEmail)) {
      this.userStates.set(userEmail, {
        currentIntent: "general",
        leaveData: {},
        conversationHistory: [],
        isCollectingLeaveData: false,
        lastMessage: null,
      });
    }
    return this.userStates.get(userEmail);
  }

  // Classify user intent
  async classifyIntent(userMessage, userEmail) {
    const systemPrompt = `You are an intent classification system for a workplace chatbot. 
    Classify the user's message into one of these intents:
    
    1. "leave_application" - User wants to apply for leave, vacation, time off, sick leave, etc.
    2. "general" - General conversation, greetings, questions, etc.
    
    Examples of leave_application:
    - "I want to take leave"
    - "Can I apply for vacation?"
    - "I need sick leave"
    - "I want time off next week"
    - "How do I request leave?"
    
    Respond with ONLY the intent name: "leave_application" or "general"`;

    try {
      const response = await this.llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage(userMessage),
      ]);

      return response.content.toLowerCase().trim();
    } catch (error) {
      console.error("Error classifying intent:", error);
      return "general";
    }
  }

  // Check if all required leave data is collected
  isLeaveDataComplete(leaveData) {
    return this.requiredFields.every(
      (field) => leaveData[field] && leaveData[field].trim() !== ""
    );
  }

  // Extract leave information from user message
  async extractLeaveInfo(userMessage, currentLeaveData) {
    const systemPrompt = `You are a data extraction system for leave applications. 
    Extract leave-related information from the user's message and return it in JSON format.
    
    Available fields to extract:
    - startDate: When the leave starts (format: YYYY-MM-DD or relative like "next Monday")
    - endDate: When the leave ends (format: YYYY-MM-DD or relative like "Friday")
    - leaveType: Type of leave (casual, sick, vacation, personal, emergency, etc.)
    - reason: Reason for the leave
    
    Current data already collected: ${JSON.stringify(currentLeaveData)}
    
    Only extract NEW information from the current message. If a field is already collected, don't overwrite it unless the user is correcting it.
    
    Return ONLY a JSON object with the extracted fields. If no new information is found, return an empty object {}.
    
    Examples:
    - "I want to take vacation from March 15 to March 20" → {"startDate": "March 15", "endDate": "March 20", "leaveType": "vacation"}
    - "I'm sick and need to take leave tomorrow" → {"startDate": "tomorrow", "leaveType": "sick"}
    - "I need personal leave next week" → {"startDate": "next week", "leaveType": "personal"}`;

    try {
      const response = await this.llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage(userMessage),
      ]);

      const extractedData = JSON.parse(response.content);
      return extractedData;
    } catch (error) {
      console.error("Error extracting leave info:", error);
      return {};
    }
  }

  // Generate conversational response
  async generateResponse(userMessage, userState) {
    const { leaveData, isCollectingLeaveData } = userState;

    if (isCollectingLeaveData) {
      // Extract new information
      const extractedData = await this.extractLeaveInfo(userMessage, leaveData);

      // Update leave data
      Object.assign(leaveData, extractedData);

      // Check if all data is complete
      if (this.isLeaveDataComplete(leaveData)) {
        userState.isCollectingLeaveData = false;
        return {
          response: `Perfect! I have all the information needed for your leave application:
          
📅 **Leave Details:**
- **Start Date:** ${leaveData.startDate}
- **End Date:** ${leaveData.endDate}
- **Leave Type:** ${leaveData.leaveType}
- **Reason:** ${leaveData.reason}

I'll now process your leave application. Please wait a moment...`,
          shouldInvokeLeaveAgent: true,
          leaveData: { ...leaveData },
        };
      } else {
        // Ask for missing information
        const missingFields = this.requiredFields.filter(
          (field) => !leaveData[field] || leaveData[field].trim() === ""
        );

        const fieldPrompts = {
          startDate: "When would you like your leave to start?",
          endDate: "When should your leave end?",
          leaveType:
            "What type of leave is this? (casual, sick, vacation, personal, etc.)",
          reason: "What's the reason for your leave?",
        };

        const nextField = missingFields[0];
        return {
          response: fieldPrompts[nextField],
          shouldInvokeLeaveAgent: false,
          leaveData: { ...leaveData },
        };
      }
    } else {
      // General conversation
      const systemPrompt = `You are a helpful workplace assistant. Be friendly, professional, and concise. 
      If the user mentions anything related to leave, vacation, time off, or sick days, gently guide them to provide more details about their leave request.`;

      try {
        const response = await this.llm.invoke([
          new SystemMessage(systemPrompt),
          new HumanMessage(userMessage),
        ]);

        return {
          response: response.content,
          shouldInvokeLeaveAgent: false,
          leaveData: null,
        };
      } catch (error) {
        console.error("Error generating response:", error);
        return {
          response: "I'm here to help! How can I assist you today?",
          shouldInvokeLeaveAgent: false,
          leaveData: null,
        };
      }
    }
  }

  // Main processing method
  async processMessage(userMessage, userEmail) {
    const userState = this.getOrCreateUserState(userEmail);

    // Classify intent
    const intent = await this.classifyIntent(userMessage, userEmail);

    // Update user state based on intent
    if (intent === "leave_application" && !userState.isCollectingLeaveData) {
      userState.isCollectingLeaveData = true;
      userState.leaveData = {};

      // Extract initial leave information
      const extractedData = await this.extractLeaveInfo(userMessage, {});
      Object.assign(userState.leaveData, extractedData);

      // If we have some data, continue collection
      if (Object.keys(userState.leaveData).length > 0) {
        const missingFields = this.requiredFields.filter(
          (field) =>
            !userState.leaveData[field] ||
            userState.leaveData[field].trim() === ""
        );

        if (missingFields.length > 0) {
          const fieldPrompts = {
            startDate: "When would you like your leave to start?",
            endDate: "When should your leave end?",
            leaveType:
              "What type of leave is this? (casual, sick, vacation, personal, etc.)",
            reason: "What's the reason for your leave?",
          };

          const nextField = missingFields[0];
          return {
            response: fieldPrompts[nextField],
            shouldInvokeLeaveAgent: false,
            leaveData: { ...userState.leaveData },
          };
        }
      } else {
        return {
          response:
            "I'd be happy to help you apply for leave! Could you tell me when you'd like to start your leave?",
          shouldInvokeLeaveAgent: false,
          leaveData: {},
        };
      }
    }

    // Generate response
    const result = await this.generateResponse(userMessage, userState);

    // Update conversation history
    userState.conversationHistory.push({
      user: userMessage,
      assistant: result.response,
      timestamp: new Date().toISOString(),
    });

    // Keep only last 10 messages to prevent memory issues
    if (userState.conversationHistory.length > 10) {
      userState.conversationHistory = userState.conversationHistory.slice(-10);
    }

    return result;
  }

  // Reset user state (useful for testing or when user wants to start over)
  resetUserState(userEmail) {
    this.userStates.delete(userEmail);
  }
}
