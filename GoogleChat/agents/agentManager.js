import { MainAgent } from "./mainAgent.js";
import { LeaveAgent } from "./leaveAgent.js";

export class AgentManager {
  constructor() {
    this.mainAgent = new MainAgent();
    this.leaveAgent = new LeaveAgent();
  }

  // Main entry point for processing user messages
  async processMessage(userMessage, userEmail) {
    try {
      console.log(`🤖 Processing message from ${userEmail}: "${userMessage}"`);

      // Process with Main Agent first
      const mainAgentResult = await this.mainAgent.processMessage(
        userMessage,
        userEmail
      );

      // If Main Agent indicates we should invoke Leave Agent
      if (mainAgentResult.shouldInvokeLeaveAgent && mainAgentResult.leaveData) {
        console.log(`📋 Leave data complete, invoking Leave Agent...`);

        // Process with Leave Agent
        const leaveAgentResult = await this.leaveAgent.processLeaveApplication(
          {
            ...mainAgentResult.leaveData,
          },
          userEmail
        );

        // Combine results
        return {
          success: leaveAgentResult.success,
          message: leaveAgentResult.message,
          applicationId: leaveAgentResult.applicationId,
          status: leaveAgentResult.status,
          conflicts: leaveAgentResult.conflicts,
          approvalStatus: leaveAgentResult.approvalStatus,
          error: leaveAgentResult.error,
        };
      } else {
        // Return Main Agent response for general conversation or incomplete leave data
        return {
          success: true,
          message: mainAgentResult.response,
          isLeaveApplication: false,
          leaveData: mainAgentResult.leaveData,
        };
      }
    } catch (error) {
      console.error("❌ Error in AgentManager:", error);
      return {
        success: false,
        message:
          "I'm sorry, I encountered an error while processing your request. Please try again.",
        error: error.message,
      };
    }
  }

  // Get application status
  getApplicationStatus(applicationId) {
    return this.leaveAgent.getApplicationStatus(applicationId);
  }

  // Reset user state (useful for testing)
  resetUserState(userEmail) {
    this.mainAgent.resetUserState(userEmail);
  }

  // Get all leave applications (for admin purposes)
  getAllApplications() {
    return this.leaveAgent.getAllApplications();
  }
}
