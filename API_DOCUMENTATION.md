# Voice Testing Platform API Documentation

## Scaled Testing Endpoints

### POST /twilio-test
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/twilio-test" \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
    "user_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/another-agent-id",
    "concurrent_requests": 5,
    "timeout": 30,
    "sample_rate": 8000,
    "encoding": "mulaw"
  }'
```

**Payload:**
```json
{
  "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
  "user_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/another-agent-id",
  "concurrent_requests": 5,
  "timeout": 30,
  "sample_rate": 8000,
  "encoding": "mulaw"
}
```

**Response:**
```json
{
  "success": true,
  "test_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Twilio test started with 5 concurrent connections",
  "status": {
    "test_id": "550e8400-e29b-41d4-a716-446655440000",
    "concurrent_requests": 5,
    "timeout": 30,
    "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
    "user_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/another-agent-id",
    "sample_rate": 8000,
    "encoding": "mulaw",
    "status": "running"
  }
}
```

### POST /web-test
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/web-test" \
  -H "Content-Type: application/json" \
  -d '{
    "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
    "user_agent_id": "028cdb34-ef3e-40d9-a4bd-35b92d9c5d29",
    "concurrent_requests": 3,
    "timeout": 45,
    "ws_url_base": "ws://localhost:5050"
  }'
```

**Payload:**
```json
{
  "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
  "user_agent_id": "028cdb34-ef3e-40d9-a4bd-35b92d9c5d29",
  "concurrent_requests": 3,
  "timeout": 45,
  "ws_url_base": "ws://localhost:5050"
}
```

**Response:**
```json
{
  "success": true,
  "test_id": "550e8400-e29b-41d4-a716-446655440001",
  "message": "Web test started with 3 concurrent connections",
  "status": {
    "test_id": "550e8400-e29b-41d4-a716-446655440001",
    "concurrent_requests": 3,
    "timeout": 45,
    "target_agent_uri": "ws://localhost:5050/api/call/media-stream/agents/d7794b6e-20ff-4e28-9a97-98e327eea2d3",
    "user_agent_id": "028cdb34-ef3e-40d9-a4bd-35b92d9c5d29",
    "ws_url_base": "ws://localhost:5050",
    "status": "running"
  }
}
```

## Test Suite CRUD Endpoints

### POST /v1/test-suites
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/v1/test-suites?user_id=12345678-1234-1234-1234-123456789012" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Test Suite",
    "description": "Testing voice agent interactions",
    "target_agent_id": "87654321-4321-4321-4321-210987654321",
    "user_agent_id": "abcd1234-5678-9012-3456-789012345678"
  }'
```

**Payload:**
```json
{
  "name": "My Test Suite",
  "description": "Testing voice agent interactions",
  "target_agent_id": "87654321-4321-4321-4321-210987654321",
  "user_agent_id": "abcd1234-5678-9012-3456-789012345678"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "My Test Suite",
  "description": "Testing voice agent interactions",
  "target_agent_id": "87654321-4321-4321-4321-210987654321",
  "user_agent_id": "abcd1234-5678-9012-3456-789012345678",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/test-suites/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-suites/550e8400-e29b-41d4-a716-446655440002"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "My Test Suite",
  "description": "Testing voice agent interactions",
  "target_agent_id": "87654321-4321-4321-4321-210987654321",
  "user_agent_id": "abcd1234-5678-9012-3456-789012345678",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/test-suites/{id}/details
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-suites/550e8400-e29b-41d4-a716-446655440002/details"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "My Test Suite",
  "description": "Testing voice agent interactions",
  "target_agent_id": "87654321-4321-4321-4321-210987654321",
  "user_agent_id": "abcd1234-5678-9012-3456-789012345678",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00",
  "target_agent": {
    "id": "87654321-4321-4321-4321-210987654321",
    "name": "Test Agent",
    "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
    "sample_rate": 8000,
    "encoding": "mulaw"
  },
  "user_agent": {
    "id": "abcd1234-5678-9012-3456-789012345678",
    "name": "User Agent",
    "system_prompt": "You are a helpful assistant",
    "evaluation_criteria": {},
    "model_config": {}
  },
  "test_cases": []
}
```

### GET /v1/test-suites
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-suites?user_id=12345678-1234-1234-1234-123456789012&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_suites": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440002",
      "user_id": "12345678-1234-1234-1234-123456789012",
      "name": "My Test Suite",
      "description": "Testing voice agent interactions",
      "target_agent_id": "87654321-4321-4321-4321-210987654321",
      "user_agent_id": "abcd1234-5678-9012-3456-789012345678",
      "created_at": "2026-01-01T13:00:00",
      "updated_at": "2026-01-01T13:00:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### PUT /v1/test-suites/{id}
**Curl Command:**
```bash
curl -X PUT "http://localhost:8080/v1/test-suites/550e8400-e29b-41d4-a716-446655440002" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Test Suite",
    "description": "Updated description"
  }'
```

**Payload:**
```json
{
  "name": "Updated Test Suite",
  "description": "Updated description"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "Updated Test Suite",
  "description": "Updated description",
  "target_agent_id": "87654321-4321-4321-4321-210987654321",
  "user_agent_id": "abcd1234-5678-9012-3456-789012345678",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:01:00"
}
```

### DELETE /v1/test-suites/{id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/v1/test-suites/550e8400-e29b-41d4-a716-446655440002"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Test suite '550e8400-e29b-41d4-a716-446655440002' deleted successfully"
}
```

## Target Agent CRUD Endpoints

### POST /v1/target-agents
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/v1/target-agents?user_id=12345678-1234-1234-1234-123456789012" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Agent",
    "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
    "sample_rate": 8000,
    "encoding": "mulaw"
  }'
```

**Payload:**
```json
{
  "name": "Test Agent",
  "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
  "sample_rate": 8000,
  "encoding": "mulaw"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "Test Agent",
  "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
  "sample_rate": 8000,
  "encoding": "mulaw",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/target-agents/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/target-agents/550e8400-e29b-41d4-a716-446655440003"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "Test Agent",
  "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
  "sample_rate": 8000,
  "encoding": "mulaw",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/target-agents
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/target-agents?user_id=12345678-1234-1234-1234-123456789012&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "target_agents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440003",
      "user_id": "12345678-1234-1234-1234-123456789012",
      "name": "Test Agent",
      "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
      "sample_rate": 8000,
      "encoding": "mulaw",
      "created_at": "2026-01-01T13:00:00",
      "updated_at": "2026-01-01T13:00:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### PUT /v1/target-agents/{id}
**Curl Command:**
```bash
curl -X PUT "http://localhost:8080/v1/target-agents/550e8400-e29b-41d4-a716-446655440003" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Agent",
    "sample_rate": 16000
  }'
```

**Payload:**
```json
{
  "name": "Updated Agent",
  "sample_rate": 16000
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440003",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "Updated Agent",
  "websocket_url": "ws://localhost:5050/api/call/media-stream/agents/test",
  "sample_rate": 16000,
  "encoding": "mulaw",
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:01:00"
}
```

### DELETE /v1/target-agents/{id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/v1/target-agents/550e8400-e29b-41d4-a716-446655440003"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Target agent '550e8400-e29b-41d4-a716-446655440003' deleted successfully"
}
```

## User Agent CRUD Endpoints

### POST /v1/user-agents
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/v1/user-agents?user_id=12345678-1234-1234-1234-123456789012" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "User Agent",
    "system_prompt": "You are a helpful assistant",
    "evaluation_criteria": {"accuracy": 0.8},
    "model_config": {"temperature": 0.7}
  }'
```

**Payload:**
```json
{
  "name": "User Agent",
  "system_prompt": "You are a helpful assistant",
  "evaluation_criteria": {"accuracy": 0.8},
  "model_config": {"temperature": 0.7}
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "User Agent",
  "system_prompt": "You are a helpful assistant",
  "evaluation_criteria": {"accuracy": 0.8},
  "model_config": {"temperature": 0.7},
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/user-agents/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/user-agents/550e8400-e29b-41d4-a716-446655440004"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "User Agent",
  "system_prompt": "You are a helpful assistant",
  "evaluation_criteria": {"accuracy": 0.8},
  "model_config": {"temperature": 0.7},
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/user-agents
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/user-agents?user_id=12345678-1234-1234-1234-123456789012&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "user_agents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440004",
      "user_id": "12345678-1234-1234-1234-123456789012",
      "name": "User Agent",
      "system_prompt": "You are a helpful assistant",
      "evaluation_criteria": {"accuracy": 0.8},
      "model_config": {"temperature": 0.7},
      "created_at": "2026-01-01T13:00:00",
      "updated_at": "2026-01-01T13:00:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### PUT /v1/user-agents/{id}
**Curl Command:**
```bash
curl -X PUT "http://localhost:8080/v1/user-agents/550e8400-e29b-41d4-a716-446655440004" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated User Agent",
    "system_prompt": "You are an advanced assistant"
  }'
```

**Payload:**
```json
{
  "name": "Updated User Agent",
  "system_prompt": "You are an advanced assistant"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440004",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "name": "Updated User Agent",
  "system_prompt": "You are an advanced assistant",
  "evaluation_criteria": {"accuracy": 0.8},
  "model_config": {"temperature": 0.7},
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:01:00"
}
```

### DELETE /v1/user-agents/{id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/v1/user-agents/550e8400-e29b-41d4-a716-446655440004"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "User agent '550e8400-e29b-41d4-a716-446655440004' deleted successfully"
}
```

## Test Case CRUD Endpoints

### POST /v1/test-cases
**Curl Command:**
```bash
curl -X POST "http://localhost:8080/v1/test-cases" \
  -H "Content-Type: application/json" \
  -d '{
    "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
    "name": "Test Case 1",
    "steps": [{"action": "speak", "text": "Hello"}],
    "conditions": [{"type": "response", "expected": "Hi there"}],
    "expected_outcome": "Agent responds appropriately",
    "timeout_seconds": 30
  }'
```

**Payload:**
```json
{
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "name": "Test Case 1",
  "steps": [{"action": "speak", "text": "Hello"}],
  "conditions": [{"type": "response", "expected": "Hi there"}],
  "expected_outcome": "Agent responds appropriately",
  "timeout_seconds": 30
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440005",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "name": "Test Case 1",
  "steps": [{"action": "speak", "text": "Hello"}],
  "conditions": [{"type": "response", "expected": "Hi there"}],
  "expected_outcome": "Agent responds appropriately",
  "timeout_seconds": 30,
  "order_index": 0,
  "is_active": true,
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/test-cases/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-cases/550e8400-e29b-41d4-a716-446655440005"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440005",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "name": "Test Case 1",
  "steps": [{"action": "speak", "text": "Hello"}],
  "conditions": [{"type": "response", "expected": "Hi there"}],
  "expected_outcome": "Agent responds appropriately",
  "timeout_seconds": 30,
  "order_index": 0,
  "is_active": true,
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:00:00"
}
```

### GET /v1/test-cases
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-cases?suite_id=550e8400-e29b-41d4-a716-446655440002&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_cases": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440005",
      "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
      "name": "Test Case 1",
      "steps": [{"action": "speak", "text": "Hello"}],
      "conditions": [{"type": "response", "expected": "Hi there"}],
      "expected_outcome": "Agent responds appropriately",
      "timeout_seconds": 30,
      "order_index": 0,
      "is_active": true,
      "created_at": "2026-01-01T13:00:00",
      "updated_at": "2026-01-01T13:00:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### PUT /v1/test-cases/{id}
**Curl Command:**
```bash
curl -X PUT "http://localhost:8080/v1/test-cases/550e8400-e29b-41d4-a716-446655440005" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Test Case",
    "timeout_seconds": 45
  }'
```

**Payload:**
```json
{
  "name": "Updated Test Case",
  "timeout_seconds": 45
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440005",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "name": "Updated Test Case",
  "steps": [{"action": "speak", "text": "Hello"}],
  "conditions": [{"type": "response", "expected": "Hi there"}],
  "expected_outcome": "Agent responds appropriately",
  "timeout_seconds": 45,
  "order_index": 0,
  "is_active": true,
  "created_at": "2026-01-01T13:00:00",
  "updated_at": "2026-01-01T13:01:00"
}
```

### PUT /v1/test-cases/reorder/{suite_id}
**Curl Command:**
```bash
curl -X PUT "http://localhost:8080/v1/test-cases/reorder/550e8400-e29b-41d4-a716-446655440002" \
  -H "Content-Type: application/json" \
  -d '{
    "case_orders": [
      {"case_id": "550e8400-e29b-41d4-a716-446655440005", "order_index": 1}
    ]
  }'
```

**Payload:**
```json
{
  "case_orders": [
    {"case_id": "550e8400-e29b-41d4-a716-446655440005", "order_index": 1}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Test cases reordered for suite '550e8400-e29b-41d4-a716-446655440002'"
}
```

### DELETE /v1/test-cases/{id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/v1/test-cases/550e8400-e29b-41d4-a716-446655440005"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Test case '550e8400-e29b-41d4-a716-446655440005' deleted successfully"
}
```

## Test History Endpoints (Read-Only)

### GET /v1/test-history/runs/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/runs/550e8400-e29b-41d4-a716-446655440006"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440006",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "status": "completed",
  "total_test_cases": 5,
  "passed_count": 4,
  "failed_count": 1,
  "alert_count": 2,
  "started_at": "2026-01-01T12:00:00",
  "completed_at": "2026-01-01T12:05:00"
}
```

### GET /v1/test-history/runs/{id}/details
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/runs/550e8400-e29b-41d4-a716-446655440006/details"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440006",
  "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
  "user_id": "12345678-1234-1234-1234-123456789012",
  "status": "completed",
  "total_test_cases": 5,
  "passed_count": 4,
  "failed_count": 1,
  "alert_count": 2,
  "started_at": "2026-01-01T12:00:00",
  "completed_at": "2026-01-01T12:05:00",
  "test_case_results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440007",
      "test_run_id": "550e8400-e29b-41d4-a716-446655440006",
      "test_case_id": "550e8400-e29b-41d4-a716-446655440005",
      "status": "pass",
      "started_at": "2026-01-01T12:00:00",
      "completed_at": "2026-01-01T12:01:00"
    }
  ]
}
```

### GET /v1/test-history/runs
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/runs?suite_id=550e8400-e29b-41d4-a716-446655440002&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_runs": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440006",
      "test_suite_id": "550e8400-e29b-41d4-a716-446655440002",
      "user_id": "12345678-1234-1234-1234-123456789012",
      "status": "completed",
      "total_test_cases": 5,
      "passed_count": 4,
      "failed_count": 1,
      "alert_count": 2,
      "started_at": "2026-01-01T12:00:00",
      "completed_at": "2026-01-01T12:05:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### GET /v1/test-history/results/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/results/550e8400-e29b-41d4-a716-446655440007"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440007",
  "test_run_id": "550e8400-e29b-41d4-a716-446655440006",
  "test_case_id": "550e8400-e29b-41d4-a716-446655440005",
  "status": "pass",
  "recording_file_id": "file-123",
  "conversation_logs": [
    {"speaker": "user", "text": "Hello"},
    {"speaker": "agent", "text": "Hi there!"}
  ],
  "evaluation_result": {"accuracy": 0.95},
  "started_at": "2026-01-01T12:00:00",
  "completed_at": "2026-01-01T12:01:00"
}
```

### GET /v1/test-history/results/{id}/details
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/results/550e8400-e29b-41d4-a716-446655440007/details"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440007",
  "test_run_id": "550e8400-e29b-41d4-a716-446655440006",
  "test_case_id": "550e8400-e29b-41d4-a716-446655440005",
  "status": "pass",
  "recording_file_id": "file-123",
  "conversation_logs": [
    {"speaker": "user", "text": "Hello"},
    {"speaker": "agent", "text": "Hi there!"}
  ],
  "evaluation_result": {"accuracy": 0.95},
  "started_at": "2026-01-01T12:00:00",
  "completed_at": "2026-01-01T12:01:00",
  "alerts": []
}
```

### GET /v1/test-history/results
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/results?run_id=550e8400-e29b-41d4-a716-446655440006&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440007",
      "test_run_id": "550e8400-e29b-41d4-a716-446655440006",
      "test_case_id": "550e8400-e29b-41d4-a716-446655440005",
      "status": "pass",
      "started_at": "2026-01-01T12:00:00",
      "completed_at": "2026-01-01T12:01:00"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### GET /v1/test-history/alerts/{id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/alerts/550e8400-e29b-41d4-a716-446655440008"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440008",
  "test_case_result_id": "550e8400-e29b-41d4-a716-446655440007",
  "alert_type": "latency",
  "severity": "medium",
  "message": "Response time exceeded threshold",
  "created_at": "2026-01-01T12:01:30"
}
```

### GET /v1/test-history/alerts
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/v1/test-history/alerts?result_id=550e8400-e29b-41d4-a716-446655440007&limit=10&offset=0"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_alerts": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440008",
      "test_case_result_id": "550e8400-e29b-41d4-a716-446655440007",
      "alert_type": "latency",
      "severity": "medium",
      "message": "Response time exceeded threshold",
      "created_at": "2026-01-01T12:01:30"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

## Test Management Endpoints

### GET /twilio-test/{test_id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/twilio-test/550e8400-e29b-41d4-a716-446655440000"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "message": "Twilio test is currently running"
}
```

### GET /twilio-tests
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/twilio-tests"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "total": 1,
  "active_tests": ["550e8400-e29b-41d4-a716-446655440000"]
}
```

### DELETE /twilio-test/{test_id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/twilio-test/550e8400-e29b-41d4-a716-446655440000"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Twilio test '550e8400-e29b-41d4-a716-446655440000' deleted"
}
```

### GET /web-test/{test_id}
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/web-test/550e8400-e29b-41d4-a716-446655440001"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "test_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "running",
  "message": "Web test is currently running"
}
```

### GET /web-tests
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/web-tests"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "total": 1,
  "active_tests": ["550e8400-e29b-41d4-a716-446655440001"]
}
```

### DELETE /web-test/{test_id}
**Curl Command:**
```bash
curl -X DELETE "http://localhost:8080/web-test/550e8400-e29b-41d4-a716-446655440001"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "success": true,
  "message": "Web test '550e8400-e29b-41d4-a716-446655440001' deleted"
}
```

## Health Check

### GET /health
**Curl Command:**
```bash
curl -X GET "http://localhost:8080/health"
```

**Payload:**
```json
{}
```

**Response:**
```json
{
  "status": "healthy",
  "service": "voice_testing_platform",
  "version": "0.1.0"
}
```
