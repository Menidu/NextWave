import { AgentManager } from "./agents/agentManager.js";
import dotenv from "dotenv";

dotenv.config();

// Test the multi-agent system
async function testAgents() {
  console.log("🧪 Testing Multi-Agent System...\n");

  const agentManager = new AgentManager();
  const testUser = "test@example.com";

  // Test 1: General conversation
  console.log("Test 1: General Conversation");
  console.log("User: Hello, how are you?");
  const response1 = await agentManager.processMessage(
    "Hello, how are you?",
    testUser
  );
  console.log(`Bot: ${response1.message}\n`);

  // Test 2: Leave application start
  console.log("Test 2: Leave Application Start");
  console.log("User: I want to take vacation next week");
  const response2 = await agentManager.processMessage(
    "I want to take vacation next week",
    testUser
  );
  console.log(`Bot: ${response2.message}\n`);

  // Test 3: Provide start date
  console.log("Test 3: Provide Start Date");
  console.log("User: Monday");
  const response3 = await agentManager.processMessage("Monday", testUser);
  console.log(`Bot: ${response3.message}\n`);

  // Test 4: Provide end date
  console.log("Test 4: Provide End Date");
  console.log("User: Friday");
  const response4 = await agentManager.processMessage("Friday", testUser);
  console.log(`Bot: ${response4.message}\n`);

  // Test 5: Provide leave type
  console.log("Test 5: Provide Leave Type");
  console.log("User: vacation");
  const response5 = await agentManager.processMessage("vacation", testUser);
  console.log(`Bot: ${response5.message}\n`);

  // Test 6: Provide reason (should trigger Leave Agent)
  console.log("Test 6: Provide Reason (Triggers Leave Agent)");
  console.log("User: Family trip");
  const response6 = await agentManager.processMessage("Family trip", testUser);
  console.log(`Bot: ${response6.message}\n`);

  if (response6.applicationId) {
    console.log(`📋 Application ID: ${response6.applicationId}`);
    console.log(`📊 Status: ${response6.status}`);
    console.log(`✅ Approval Status: ${response6.approvalStatus}`);
  }

  // Test 7: Check application status
  if (response6.applicationId) {
    console.log("\nTest 7: Check Application Status");
    const status = agentManager.getApplicationStatus(response6.applicationId);
    console.log("Application Status:", JSON.stringify(status, null, 2));
  }

  // Test 8: Reset user state
  console.log("\nTest 8: Reset User State");
  agentManager.resetUserState(testUser);
  console.log("User state reset successfully");

  // Test 9: Get all applications
  console.log("\nTest 9: Get All Applications");
  const allApplications = agentManager.getAllApplications();
  console.log(`Total applications: ${allApplications.length}`);
  if (allApplications.length > 0) {
    console.log("Applications:", JSON.stringify(allApplications, null, 2));
  }

  console.log("\n✅ All tests completed!");
}

// Run tests if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testAgents().catch(console.error);
}

export { testAgents };
