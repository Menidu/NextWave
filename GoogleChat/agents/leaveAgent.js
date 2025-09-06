import { StateGraph, END } from "langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

export class LeaveAgent {
  constructor() {
    this.llm = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      apiKey: process.env.GEMINI_API_KEY,
    });

    // In-memory storage for leave applications (in production, use a database)
    this.leaveApplications = new Map();

    // Build the LangGraph workflow
    this.workflow = this.buildWorkflow();
  }

  // Define the state structure for the workflow
  getStateDefinition() {
    return {
      leaveData: {
        value: (x, y) => y ?? x,
        default: () => ({}),
      },
      applicationId: {
        value: (x, y) => y ?? x,
        default: () => null,
      },
      status: {
        value: (x, y) => y ?? x,
        default: () => "pending",
      },
      conflicts: {
        value: (x, y) => y ?? x,
        default: () => [],
      },
      approvalStatus: {
        value: (x, y) => y ?? x,
        default: () => "pending",
      },
      finalMessage: {
        value: (x, y) => y ?? x,
        default: () => "",
      },
      error: {
        value: (x, y) => y ?? x,
        default: () => null,
      },
    };
  }

  // Step 1: Validate and create leave application
  async validateAndCreateApplication(state) {
    try {
      const { leaveData } = state;

      // Generate unique application ID
      const applicationId = uuidv4();

      // Validate required fields
      const requiredFields = ["startDate", "endDate", "leaveType", "reason"];
      const missingFields = requiredFields.filter(
        (field) => !leaveData[field] || leaveData[field].trim() === ""
      );

      if (missingFields.length > 0) {
        return {
          ...state,
          error: `Missing required fields: ${missingFields.join(", ")}`,
          finalMessage:
            "I'm sorry, but some required information is missing from your leave application. Please try again.",
          status: "failed",
        };
      }

      // Create leave application record
      const application = {
        id: applicationId,
        ...leaveData,
        submittedAt: new Date().toISOString(),
        status: "pending",
      };

      this.leaveApplications.set(applicationId, application);

      return {
        ...state,
        applicationId,
        status: "created",
      };
    } catch (error) {
      console.error("Error in validateAndCreateApplication:", error);
      return {
        ...state,
        error: error.message,
        finalMessage:
          "An error occurred while processing your leave application. Please try again.",
        status: "failed",
      };
    }
  }

  // Step 2: Check for calendar conflicts
  async checkCalendarConflicts(state) {
    try {
      const { leaveData, applicationId } = state;

      // Simulate calendar conflict checking
      // In a real implementation, this would integrate with Google Calendar API
      const conflicts = await this.simulateCalendarCheck(leaveData);

      return {
        ...state,
        conflicts,
        status: conflicts.length > 0 ? "conflicts_found" : "no_conflicts",
      };
    } catch (error) {
      console.error("Error in checkCalendarConflicts:", error);
      return {
        ...state,
        error: error.message,
        finalMessage:
          "An error occurred while checking for calendar conflicts.",
        status: "failed",
      };
    }
  }

  // Step 3: Process approval workflow
  async processApprovalWorkflow(state) {
    try {
      const { leaveData, conflicts, applicationId } = state;

      if (conflicts.length > 0) {
        return {
          ...state,
          approvalStatus: "requires_manual_review",
          finalMessage: `⚠️ **Calendar Conflicts Detected**

Your leave application has been submitted, but there are some calendar conflicts that need manual review:

${conflicts.map((conflict) => `- ${conflict}`).join("\n")}

Your application ID is: \`${applicationId}\`

A manager will review your request and get back to you within 24 hours.`,
          status: "completed",
        };
      }

      // Simulate approval process
      const approvalResult = await this.simulateApprovalProcess(leaveData);

      // Update application status
      const application = this.leaveApplications.get(applicationId);
      if (application) {
        application.approvalStatus = approvalResult.status;
        application.approvedAt = new Date().toISOString();
        this.leaveApplications.set(applicationId, application);
      }

      return {
        ...state,
        approvalStatus: approvalResult.status,
        finalMessage: approvalResult.message,
        status: "completed",
      };
    } catch (error) {
      console.error("Error in processApprovalWorkflow:", error);
      return {
        ...state,
        error: error.message,
        finalMessage: "An error occurred during the approval process.",
        status: "failed",
      };
    }
  }

  // Simulate calendar conflict checking
  async simulateCalendarCheck(leaveData) {
    // Simulate some calendar conflicts based on leave type and duration
    const conflicts = [];

    // Simulate conflicts for vacation leave
    if (leaveData.leaveType.toLowerCase().includes("vacation")) {
      const startDate = new Date(leaveData.startDate);
      const endDate = new Date(leaveData.endDate);
      const duration = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));

      // Simulate conflict for long vacations
      if (duration > 7) {
        conflicts.push(
          "Long vacation period overlaps with team meeting on " +
            new Date(
              startDate.getTime() + 3 * 24 * 60 * 60 * 1000
            ).toLocaleDateString()
        );
      }
    }

    // Simulate conflicts for sick leave
    if (leaveData.leaveType.toLowerCase().includes("sick")) {
      // Simulate conflict for sick leave during important periods
      const startDate = new Date(leaveData.startDate);
      const month = startDate.getMonth();
      const day = startDate.getDate();

      // Simulate conflict during month-end (common for sick leave)
      if (day >= 25 && day <= 31) {
        conflicts.push("Sick leave during month-end reporting period");
      }
    }

    return conflicts;
  }

  // Simulate approval process
  async simulateApprovalProcess(leaveData) {
    // Simulate different approval outcomes based on leave type and duration
    const startDate = new Date(leaveData.startDate);
    const endDate = new Date(leaveData.endDate);
    const duration = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));

    // Auto-approve short casual leaves
    if (leaveData.leaveType.toLowerCase().includes("casual") && duration <= 2) {
      return {
        status: "approved",
        message: `✅ **Leave Application Approved!**

📅 **Leave Details:**
- **Start Date:** ${leaveData.startDate}
- **End Date:** ${leaveData.endDate}
- **Leave Type:** ${leaveData.leaveType}
- **Duration:** ${duration} day(s)
- **Reason:** ${leaveData.reason}

Your leave has been automatically approved. Enjoy your time off!`,
      };
    }

    // Auto-approve sick leave
    if (leaveData.leaveType.toLowerCase().includes("sick")) {
      return {
        status: "approved",
        message: `✅ **Sick Leave Approved!**

📅 **Leave Details:**
- **Start Date:** ${leaveData.startDate}
- **End Date:** ${leaveData.endDate}
- **Leave Type:** ${leaveData.leaveType}
- **Duration:** ${duration} day(s)
- **Reason:** ${leaveData.reason}

Your sick leave has been approved. Take care and get well soon!`,
      };
    }

    // Require manual approval for other types
    return {
      status: "pending_approval",
      message: `⏳ **Leave Application Submitted**

📅 **Leave Details:**
- **Start Date:** ${leaveData.startDate}
- **End Date:** ${leaveData.endDate}
- **Leave Type:** ${leaveData.leaveType}
- **Duration:** ${duration} day(s)
- **Reason:** ${leaveData.reason}

Your leave application has been submitted and is pending manager approval. You'll receive a notification once it's reviewed.`,
    };
  }

  // Build the LangGraph workflow
  buildWorkflow() {
    const workflow = new StateGraph(this.getStateDefinition());

    // Add nodes
    workflow.addNode("validate", this.validateAndCreateApplication.bind(this));
    workflow.addNode("check_conflicts", this.checkCalendarConflicts.bind(this));
    workflow.addNode(
      "process_approval",
      this.processApprovalWorkflow.bind(this)
    );

    // Add edges
    workflow.addEdge("validate", "check_conflicts");
    workflow.addEdge("check_conflicts", "process_approval");
    workflow.addEdge("process_approval", END);

    // Set entry point
    workflow.setEntryPoint("validate");

    return workflow.compile();
  }

  // Main method to process leave application
  async processLeaveApplication(leaveData) {
    try {
      const initialState = {
        leaveData,
        applicationId: null,
        status: "pending",
        conflicts: [],
        approvalStatus: "pending",
        finalMessage: "",
        error: null,
      };

      // Execute the workflow
      const result = await this.workflow.invoke(initialState);

      return {
        success: !result.error,
        message: result.finalMessage,
        applicationId: result.applicationId,
        status: result.status,
        conflicts: result.conflicts,
        approvalStatus: result.approvalStatus,
        error: result.error,
      };
    } catch (error) {
      console.error("Error processing leave application:", error);
      return {
        success: false,
        message:
          "An unexpected error occurred while processing your leave application. Please try again.",
        error: error.message,
      };
    }
  }

  // Get application status
  getApplicationStatus(applicationId) {
    const application = this.leaveApplications.get(applicationId);
    if (!application) {
      return null;
    }

    return {
      id: application.id,
      status: application.status,
      approvalStatus: application.approvalStatus,
      submittedAt: application.submittedAt,
      approvedAt: application.approvedAt,
      ...application,
    };
  }

  // List all applications (for admin purposes)
  getAllApplications() {
    return Array.from(this.leaveApplications.values());
  }
}
