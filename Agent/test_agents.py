import asyncio
import json
from dotenv import load_dotenv
from agents.agent_manager import AgentManager

# Load environment variables
load_dotenv()

async def test_agents():
    """Test the multi-agent system"""
    print("🧪 Testing Multi-Agent System...\n")
    
    agent_manager = AgentManager()
    test_user = "test@example.com"
    
    # Test 1: General conversation
    print("Test 1: General Conversation")
    print("User: Hello, how are you?")
    response1 = await agent_manager.process_message("Hello, how are you?", test_user)
    print(f"Bot: {response1['message']}\n")
    
    # Test 2: Leave application start
    print("Test 2: Leave Application Start")
    print("User: I want to take vacation next week")
    response2 = await agent_manager.process_message("I want to take vacation next week", test_user)
    print(f"Bot: {response2['message']}\n")
    
    # Test 3: Provide start date
    print("Test 3: Provide Start Date")
    print("User: Monday")
    response3 = await agent_manager.process_message("Monday", test_user)
    print(f"Bot: {response3['message']}\n")
    
    # Test 4: Provide end date
    print("Test 4: Provide End Date")
    print("User: Friday")
    response4 = await agent_manager.process_message("Friday", test_user)
    print(f"Bot: {response4['message']}\n")
    
    # Test 5: Provide leave type
    print("Test 5: Provide Leave Type")
    print("User: vacation")
    response5 = await agent_manager.process_message("vacation", test_user)
    print(f"Bot: {response5['message']}\n")
    
    # Test 6: Provide reason (should trigger Leave Agent)
    print("Test 6: Provide Reason (Triggers Leave Agent)")
    print("User: Family trip")
    response6 = await agent_manager.process_message("Family trip", test_user)
    print(f"Bot: {response6['message']}\n")
    
    if response6.get("applicationId"):
        print(f"📋 Application ID: {response6['applicationId']}")
        print(f"📊 Status: {response6.get('status')}")
        print(f"✅ Approval Status: {response6.get('approvalStatus')}")
    
    # Test 7: Check application status
    if response6.get("applicationId"):
        print("\nTest 7: Check Application Status")
        status = agent_manager.get_application_status(response6["applicationId"])
        print("Application Status:", json.dumps(status, indent=2, default=str))
    
    # Test 8: Reset user state
    print("\nTest 8: Reset User State")
    agent_manager.reset_user_state(test_user)
    print("User state reset successfully")
    
    # Test 9: Get all applications
    print("\nTest 9: Get All Applications")
    all_applications = agent_manager.get_all_applications()
    print(f"Total applications: {len(all_applications)}")
    if all_applications:
        print("Applications:", json.dumps(all_applications, indent=2, default=str))
    
    print("\n✅ All tests completed!")

# Test with complete leave application in one message
async def test_complete_leave_application():
    """Test complete leave application in one message"""
    print("\n🧪 Testing Complete Leave Application in One Message...\n")
    
    agent_manager = AgentManager()
    test_user = "complete_test@example.com"
    
    # Complete leave application
    print("User: I want to take sick leave from 2024-01-15 to 2024-01-17 because I have flu")
    response = await agent_manager.process_message(
        "I want to take sick leave from 2024-01-15 to 2024-01-17 because I have flu", 
        test_user
    )
    print(f"Bot: {response['message']}\n")
    
    if response.get("applicationId"):
        print(f"📋 Application ID: {response['applicationId']}")
        print(f"📊 Status: {response.get('status')}")
        print(f"✅ Approval Status: {response.get('approvalStatus')}")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_agents())
    asyncio.run(test_complete_leave_application())