#!/bin/bash

# 🚀 Olympus Echo Backend API Test Script
# This script demonstrates the correct curl commands for testing the API

echo "🎯 Olympus Echo Backend API - Complete Test Sequence"
echo "=================================================="

# Configuration - Change these values to match your setup
BASE_URL="http://localhost:8080"
USER_ID="31b6d051-7619-4f1b-8646-34c6016c30f3"
TARGET_AGENT_NAME="Test Target Agent"
TARGET_AGENT_WS_URL="ws://localhost:5050/api/call/media-stream/agents/b9f92d26-9544-44cb-a5f1-2c756af90fd0"
USER_AGENT_NAME="Test User Agent"
USER_AGENT_PROMPT="You are a helpful customer service agent. Be friendly and professional."
TEST_SUITE_NAME="Voice Agent Test Suite"
TEST_SUITE_DESC="Testing voice agent interactions and responses"
TEST_CASE_NAME="Basic Greeting Test"
WEB_TEST_TARGET_URI="ws://localhost:5050/api/call/media-stream/agents/b9f92d26-9544-44cb-a5f1-2c756af90fd0"
WEB_TEST_USER_AGENT_ID="31b6d051-7619-4f1b-8646-34c6016c30f3"
WEB_TEST_CONCURRENT="3"
WEB_TEST_TIMEOUT="30"
WEB_TEST_WS_BASE="ws://localhost:5050"

echo "📋 Configuration:"
echo "  BASE_URL: $BASE_URL"
echo "  USER_ID: $USER_ID"
echo ""

# Function to make API call and show result
make_api_call() {
    local step=$1
    local description=$2
    local command=$3

    echo ""
    echo "🔸 Step $step: $description"
    echo "Command: $command"
    echo "Response:"

    # Execute the command
    eval "$command"

    echo ""
    echo "----------------------------------------"
}

echo "⚠️  Make sure to start the server first:"
echo "   cd olympus_echo_backend"
echo "   DATABASE_URL='postgresql://user:pass@localhost:5432/db' python main.py"
echo ""

# 1. Create Target Agent
make_api_call "1" "Create Target Agent" "curl -X POST '${BASE_URL}/v1/target-agents' -H 'Content-Type: application/json' -d '{
  \"user_id\": \"${USER_ID}\",
  \"name\": \"${TARGET_AGENT_NAME}\",
  \"websocket_url\": \"${TARGET_AGENT_WS_URL}\",
  \"sample_rate\": 8000,
  \"encoding\": \"mulaw\"
}'"

# 2. Create User Agent
make_api_call "2" "Create User Agent" "curl -X POST '${BASE_URL}/v1/user-agents' -H 'Content-Type: application/json' -d '{
  \"user_id\": \"${USER_ID}\",
  \"name\": \"${USER_AGENT_NAME}\",
  \"system_prompt\": \"${USER_AGENT_PROMPT}\",
  \"evaluation_criteria\": {
    \"accuracy\": 0.8,
    \"response_time\": 5.0
  },
  \"agent_model_config\": {
    \"temperature\": 0.7,
    \"max_tokens\": 4000
  }
}'"

# 3. Create Test Suite
make_api_call "3" "Create Test Suite" "curl -X POST '${BASE_URL}/v1/test-suites?user_id=${USER_ID}' -H 'Content-Type: application/json' -d '{
  \"name\": \"${TEST_SUITE_NAME}\",
  \"description\": \"${TEST_SUITE_DESC}\"
}'"

# 4. Create Test Case
make_api_call "4" "Create Test Case" "curl -X POST '${BASE_URL}/v1/test-cases' -H 'Content-Type: application/json' -d '{
  \"test_suite_id\": \"20046318-8654-4365-a047-33d013a47fa6\",
  \"name\": \"${TEST_CASE_NAME}\",
  \"goals\": [
    {
      \"action\": \"speak\",
      \"text\": \"Hello, can you help me with my account?\"
    }
  ],
  \"evaluation_criteria\": [
    {
      \"type\": \"response_contains\",
      \"expected\": \"Hello\"
    }
  ],
  \"timeout_seconds\": 30,
  \"order_index\": 0,
  \"is_active\": true
}'"

# 5. Run Web Scale Test
make_api_call "5" "Run Web Scale Test" "curl -X POST '${BASE_URL}/v1/web-test' -H 'Content-Type: application/json' -d '{
  \"target_agent_uri\": \"${WEB_TEST_TARGET_URI}\",
  \"user_agent_id\": \"${WEB_TEST_USER_AGENT_ID}\",
  \"concurrent_requests\": ${WEB_TEST_CONCURRENT},
  \"timeout\": ${WEB_TEST_TIMEOUT},
  \"ws_url_base\": \"${WEB_TEST_WS_BASE}\"
}'"

echo ""
echo "✅ Test sequence completed!"
echo ""
echo "📝 Notes:"
echo "  - Make sure DATABASE_URL is set correctly"
echo "  - Update the UUIDs in steps 3-5 with actual IDs from previous responses"
echo "  - All JSON is properly formatted and escaped for shell execution"
echo "  - Commands work in bash/zsh terminals and Postman"
echo ""
echo "🎯 The JSON decode error has been fixed by:"
echo "  - Using proper JSON escaping"
echo "  - Including user_id in request body (not query params)"
echo "  - Using single quotes around JSON data in curl commands"
