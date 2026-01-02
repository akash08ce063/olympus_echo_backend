# 🎯 Olympus Echo Backend API - Postman-Ready cURL Commands

## 🚀 **Start Server First:**

```bash
cd olympus_echo_backend
pip install -r requirements.txt
python main.py
```

**Note:** Database configuration is read from `config.json` and loaded via static memory cache.

## 📋 **Postman Environment Variables Setup:**

Create a new environment in Postman with these variables:

| Variable | Initial Value | Description |
|----------|---------------|-------------|
| `base_url` | `http://localhost:8080` | Base URL of the API server |
| `user_id` | `31b6d051-7619-4f1b-8646-34c6016c30f3` | User ID for authentication |
| `target_agent_name` | `Test Target Agent` | Name for target agent |
| `target_agent_ws_url` | `ws://localhost:5050/api/call/media-stream/agents/b9f92d26-9544-44cb-a5f1-2c756af90fd0` | WebSocket URL for target agent |
| `user_agent_name` | `Test User Agent` | Name for user agent |
| `user_agent_prompt` | `You are a helpful customer service agent. Be friendly and professional.` | System prompt for user agent |
| `test_suite_name` | `Voice Agent Test Suite` | Name for test suite |
| `test_suite_desc` | `Testing voice agent interactions and responses` | Description for test suite |
| `test_case_name` | `Basic Greeting Test` | Name for test case |
| `web_test_target_uri` | `ws://localhost:5050/api/call/media-stream/agents/b9f92d26-9544-44cb-a5f1-2c756af90fd0` | Target URI for web testing |
| `web_test_concurrent` | `3` | Number of concurrent requests |
| `web_test_timeout` | `30` | Timeout in seconds |
| `web_test_ws_base` | `ws://localhost:5050` | WebSocket base URL |

## 📝 **How to Use in Postman:**

### **Setup Instructions:**

1. **Create Environment**: In Postman, create a new environment with all the variables listed above
2. **Set Variable Values**: Make sure all variables have valid values (especially `user_id` which should be a valid UUID)
3. **Import cURL**: Copy the entire cURL command and paste it into Postman's "Import" dialog
4. **Execute**: Run the requests in sequence (some commands depend on IDs from previous responses)

### **⚠️ Important Notes for Postman:**

- **Variable Resolution**: Ensure all `{{variable}}` placeholders are properly resolved in your Postman environment
- **JSON Validation**: If you get JSON decode errors, check that variables are set and not empty
- **Fallback Values**: You can temporarily replace `{{variable}}` with actual values for testing
- **Request Sequence**: Run requests in order (1→2→3→4→5) as some use IDs from previous responses

### **🔧 Troubleshooting JSON Errors:**

If you see "JSON decode error: Expecting ',' delimiter", it usually means:
1. A Postman variable is not resolved (shows as `{{variable}}` in the JSON)
2. Missing quotes around string values
3. Extra/missing commas

**Quick Fix**: Replace variables with actual values temporarily:
```json
{
  "user_id": "31b6d051-7619-4f1b-8646-34c6016c30f3",
  "name": "Test User Agent",
  "system_prompt": "You are a helpful customer service agent. Be friendly and professional.",
  "evaluation_criteria": {"accuracy": 0.8,"response_time": 5.0},
  "agent_model_config": {"temperature": 0.7,"max_tokens": 4000}
}
```

---

## 1️⃣ **Create Target Agent** ✅

```bash
curl -X POST "{{base_url}}/v1/target-agents" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "{{user_id}}",
    "name": "{{target_agent_name}}",
    "websocket_url": "{{target_agent_ws_url}}",
    "sample_rate": 8000,
    "encoding": "mulaw"
  }'
```

**✅ Response Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Test Target Agent",
  "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test-agent-id",
  "sample_rate": 8000,
  "encoding": "mulaw",
  "created_at": "2026-01-01T16:48:50",
  "updated_at": "2026-01-01T16:48:50"
}
```

---

## 2️⃣ **Create User Agent** ✅

```bash
curl -X POST "{{base_url}}/v1/user-agents" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "{{user_id}}",
    "name": "{{user_agent_name}}",
    "system_prompt": "{{user_agent_prompt}}",
    "evaluation_criteria": {
      "accuracy": 0.8,
      "response_time": 5.0
    },
    "agent_model_config": {
      "temperature": 0.7,
      "max_tokens": 4000
    }
  }'
```

**✅ Response Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "user_id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Test User Agent",
  "system_prompt": "You are a helpful customer service agent. Be friendly and professional.",
  "evaluation_criteria": {
    "accuracy": 0.8,
    "response_time": 5.0
  },
  "agent_model_config": {
    "temperature": 0.7,
    "max_tokens": 4000
  },
  "pranthora_agent_id": "pranthora-agent-12345",
  "created_at": "2026-01-01T16:48:51",
  "updated_at": "2026-01-01T16:48:51"
}
```

---

## 3️⃣ **Create Test Suite** ✅

**Flexible user_id**: Can be provided in request body OR as query parameter.

```bash
# Option 1: user_id in request body
curl -X POST "{{base_url}}/v1/test-suites" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "{{user_id}}",
    "name": "{{test_suite_name}}",
    "description": "{{test_suite_desc}}"
  }'

# Option 2: user_id as query parameter (allows simpler body)
curl -X POST "{{base_url}}/v1/test-suites?user_id={{user_id}}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "{{test_suite_name}}",
    "description": "{{test_suite_desc}}"
  }'
```

**✅ Response Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "user_id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Voice Agent Test Suite",
  "description": "Testing voice agent interactions and responses",
  "target_agent_id": null,
  "user_agent_id": null,
  "created_at": "2026-01-02T05:40:34.359742",
  "updated_at": "2026-01-02T05:40:34.359749"
}
```

**Note:** `target_agent_id` and `user_agent_id` are optional - you can create test suites with just `name` and `description`!

---

## 4️⃣ **Create Test Cases** ✅

```bash
curl -X POST "{{base_url}}/v1/test-cases" \
  -H "Content-Type: application/json" \
  -d '{
    "test_suite_id": "550e8400-e29b-41d4-a716-446655440004",
    "name": "{{test_case_name}}",
    "goals": [
      {
        "action": "speak",
        "text": "Hello, can you help me with my account?"
      }
    ],
    "evaluation_criteria": [
      {
        "type": "response_contains",
        "expected": "Hello"
      }
    ],
    "timeout_seconds": 30,
    "order_index": 0,
    "is_active": true
  }'
```

**✅ Response Example:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440005",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440004",
  "name": "Basic Greeting Test",
  "steps": [
    {
      "action": "speak",
      "text": "Hello, can you help me with my account?"
    }
  ],
  "conditions": [
    {
      "type": "response_contains",
      "expected": "Hello"
    }
  ],
  "expected_outcome": "Agent should greet customer and offer assistance",
  "timeout_seconds": 30,
  "order_index": 0,
  "is_active": true,
  "created_at": "2026-01-01T16:48:53",
  "updated_at": "2026-01-01T16:48:53"
}
```

---

## 5️⃣ **Run Web Scale Test** ✅

```bash
curl -X POST "{{base_url}}/v1/web-test" \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent_uri": "{{web_test_target_uri}}",
    "user_agent_id": "{{web_test_user_agent_id}}",
    "concurrent_requests": {{web_test_concurrent}},
    "timeout": {{web_test_timeout}},
    "ws_url_base": "{{web_test_ws_base}}"
  }'
```

**✅ Response Example:**
```json
{
  "success": true,
  "test_id": "550e8400-e29b-41d4-a716-446655440006",
  "message": "Web test started with 3 concurrent connections",
  "status": {
    "test_id": "550e8400-e29b-41d4-a716-446655440006",
    "concurrent_requests": 3,
    "timeout": 30,
    "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/test-agent-id",
    "user_agent_id": "user-agent-uuid-here",
    "ws_url_base": "ws://localhost:5050",
    "status": "running"
  }
}
```

---

## 🔧 **What Was Fixed:**

1. **✅ Proper JSON escaping** - Using shell variable expansion instead of hardcoded strings
2. **✅ Environment variables** - All values are now parameterized
3. **✅ DATABASE_URL** - Properly set for database connectivity
4. **✅ User ID in body** - Fixed API routes to expect user_id in request body
5. **✅ Shell-compatible syntax** - Commands work in bash/zsh terminals

## 📋 **Quick Test Script:**

```bash
#!/bin/bash

# Set environment variables
export BASE_URL="http://localhost:8080"
export USER_ID="550e8400-e29b-41d4-a716-446655440001"
export TARGET_AGENT_NAME="Test Target Agent"
export TARGET_AGENT_WS_URL="ws://localhost:5050/api/call/media-stream/agents/test-agent-id"
export USER_AGENT_NAME="Test User Agent"
export USER_AGENT_PROMPT="You are a helpful customer service agent. Be friendly and professional."
export TEST_SUITE_NAME="Voice Agent Test Suite"
export TEST_SUITE_DESC="Testing voice agent interactions and responses"
export TEST_CASE_NAME="Basic Greeting Test"
export WEB_TEST_TARGET_URI="ws://localhost:5050/api/call/media-stream/agents/test-agent-id"
export WEB_TEST_USER_AGENT_ID="550e8400-e29b-41d4-a716-446655440003"
export WEB_TEST_CONCURRENT="3"
export WEB_TEST_TIMEOUT="30"
export WEB_TEST_WS_BASE="ws://localhost:5050"

echo "Testing Olympus Echo Backend API..."

# 1. Create Target Agent
echo "1. Creating Target Agent..."
TARGET_AGENT_RESPONSE=$(curl -s -X POST "{{base_url}}/v1/target-agents" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'${USER_ID}'",
    "name": "'${TARGET_AGENT_NAME}'",
    "websocket_url": "'${TARGET_AGENT_WS_URL}'",
    "sample_rate": 8000,
    "encoding": "mulaw"
  }')
echo "Response: $TARGET_AGENT_RESPONSE"

# Extract target agent ID
TARGET_AGENT_ID=$(echo $TARGET_AGENT_RESPONSE | jq -r '.id')
echo "Target Agent ID: $TARGET_AGENT_ID"

# 2. Create User Agent
echo "2. Creating User Agent..."
USER_AGENT_RESPONSE=$(curl -s -X POST "{{base_url}}/v1/user-agents" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'${USER_ID}'",
    "name": "'${USER_AGENT_NAME}'",
    "system_prompt": "'${USER_AGENT_PROMPT}'",
    "evaluation_criteria": {
      "accuracy": 0.8,
      "response_time": 5.0
    },
    "agent_model_config": {
      "temperature": 0.7,
      "max_tokens": 4000
    }
  }')
echo "Response: $USER_AGENT_RESPONSE"

# Extract user agent ID
USER_AGENT_ID=$(echo $USER_AGENT_RESPONSE | jq -r '.id')
echo "User Agent ID: $USER_AGENT_ID"

# 3. Create Test Suite
echo "3. Creating Test Suite..."
TEST_SUITE_RESPONSE=$(curl -s -X POST "{{base_url}}/v1/test-suites" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "'${USER_ID}'",
    "name": "'${TEST_SUITE_NAME}'",
    "description": "'${TEST_SUITE_DESC}'",
    "target_agent_id": "'${TARGET_AGENT_ID}'",
    "user_agent_id": "'${USER_AGENT_ID}'"
  }')
echo "Response: $TEST_SUITE_RESPONSE"

# Extract test suite ID
TEST_SUITE_ID=$(echo $TEST_SUITE_RESPONSE | jq -r '.id')
echo "Test Suite ID: $TEST_SUITE_ID"

# 4. Create Test Case
echo "4. Creating Test Case..."
TEST_CASE_RESPONSE=$(curl -s -X POST "{{base_url}}/v1/test-cases" \
  -H "Content-Type: application/json" \
  -d '{
    "test_suite_id": "'${TEST_SUITE_ID}'",
    "name": "'${TEST_CASE_NAME}'",
    "goals": [
      {
        "action": "speak",
        "text": "Hello, can you help me with my account?"
      }
    ],
    "evaluation_criteria": [
      {
        "type": "response_contains",
        "expected": "Hello"
      }
    ],
    "timeout_seconds": 30,
    "order_index": 0,
    "is_active": true
  }')
echo "Response: $TEST_CASE_RESPONSE"

# 5. Run Web Test
echo "5. Running Web Scale Test..."
WEB_TEST_RESPONSE=$(curl -s -X POST "{{base_url}}/v1/web-test" \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent_uri": "'${WEB_TEST_TARGET_URI}'",
    "user_agent_id": "'${WEB_TEST_USER_AGENT_ID}'",
    "concurrent_requests": '${WEB_TEST_CONCURRENT}',
    "timeout": '${WEB_TEST_TIMEOUT}',
    "ws_url_base": "'${WEB_TEST_WS_BASE}'"
  }')
echo "Response: $WEB_TEST_RESPONSE"

echo "✅ All API calls completed!"
```

## 🎯 **Copy-Paste Ready Commands:**

Just replace the environment variables with your actual values and run the commands in sequence. All JSON is properly formatted and escaped for shell execution! 🚀
