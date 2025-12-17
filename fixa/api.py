import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import ngrok
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fixa.evaluators import LocalEvaluator
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from fixa import Agent, Evaluation, Scenario, Test, TestRunner

# Load environment variables
load_dotenv(override=True)

# Suppress noisy Deepgram SDK logs
logging.getLogger("deepgram").setLevel(logging.CRITICAL)


# Pydantic Models
class AgentModel(BaseModel):
    name: str
    prompt: str
    voice_id: Optional[str] = None


class EvaluationModel(BaseModel):
    name: str
    prompt: str


class ScenarioModel(BaseModel):
    name: str
    prompt: str
    evaluations: List[EvaluationModel] = []


class TestConfig(BaseModel):
    phone_number_to_call: str
    twilio_phone_number: Optional[str] = None
    agents: List[AgentModel] = []
    scenarios: List[ScenarioModel] = []


class TestResultModel(BaseModel):
    agent: str
    scenario: str
    passed: bool
    transcript: str
    recording_url: str
    error: Optional[str] = None
    evaluations: Dict[str, Any] = {}


class TestResponse(BaseModel):
    results: List[TestResultModel]
    passed: bool


DEFAULT_VOICE_ID = "b7d50908-b17c-442d-ad8d-810c63997ed9"


# Log Streaming Setup
log_queue: asyncio.Queue = asyncio.Queue()


def sink(message):
    """
    Loguru sink that puts log records into an async queue.
    The message object from loguru is a string-like object with .record dict.
    """
    try:
        # We can put the raw string or structured data
        asyncio.create_task(log_queue.put(str(message)))
    except Exception:
        # Fallback if loop isn't running or other error
        pass


# Configure loguru to use our sink (broadcasts logs to the queue)
logger.add(sink, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# Lifespan Manager for persistent Infrastructure


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize ngrok
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if not ngrok_token:
        print("WARNING: NGROK_AUTH_TOKEN not found in environment.")

    port = 8765
    print(f"Starting ngrok tunnel on port {port}...")
    try:
        # ngrok.forward returns a listener object.
        # Using type ignore as linter sometimes flags await on Listener
        listener = await ngrok.forward(port, authtoken=ngrok_token)  # type: ignore
        app.state.ngrok_url = listener.url()
        app.state.port = port
        print(f"Ngrok tunnel established: {app.state.ngrok_url}")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")
        # We don't exit here so the server can still start, strictly speaking,
        # but tests will likely fail if they depend on ngrok.
        app.state.ngrok_url = None

    yield

    # Shutdown: Cleanup if necessary
    # ngrok-python usually handles cleanup on process exit,
    # but we can explicitly disconnect if needed.
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


def make_serializable(res: Any) -> TestResultModel:
    """
    Helper to safely convert a Fixa TestResult object into our Pydantic model.
    Handles potential missing attributes gracefully.
    """
    # 1. Try .model_dump() or .dict() if available (best case)
    if hasattr(res, "model_dump"):
        # If it matches our schema exactly, this is great.
        # But unrelated schemas might cause validation errors if we just cast blindly.
        # Ideally Fixa results map closely.
        # For now, let's extract manually to be safe like the original script did,
        # but cleaner.
        pass

    # Safe extraction
    # The result object might wrap 'test' which contains agent/scenario
    test_obj = getattr(res, "test", None)

    agent_obj = getattr(res, "agent", None)
    if not agent_obj and test_obj:
        agent_obj = getattr(test_obj, "agent", None)
    agent_name = getattr(agent_obj, "name", "Unknown") if agent_obj else "Unknown"

    scenario_obj = getattr(res, "scenario", None)
    if not scenario_obj and test_obj:
        scenario_obj = getattr(test_obj, "scenario", None)
    scenario_name = getattr(scenario_obj, "name", "Unknown") if scenario_obj else "Unknown"

    transcript = getattr(res, "transcript", "")
    if isinstance(transcript, list):
        # Convert list of dicts (e.g. [{'role': '...', 'content': '...'}]) to string
        formatted = []
        for msg in transcript:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        transcript = "\n".join(formatted)
    elif not transcript:
        transcript = ""

    return TestResultModel(
        agent=agent_name,
        scenario=scenario_name,
        passed=getattr(res, "passed", False),
        transcript=transcript,
        recording_url=getattr(res, "recording_url", "") or "",
        error=getattr(res, "error", None),
        evaluations=getattr(res, "evaluations", {}) or {},
    )


@app.get("/logs")
async def logs(request: Request):
    """
    SSE endpoint to stream logs to the client.
    """

    async def event_generator():
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Wait for log message
            message = await log_queue.get()
            yield {"data": message}

    return EventSourceResponse(event_generator())


@app.post("/test", response_model=TestResponse)
async def run_test(config: TestConfig):
    if not app.state.ngrok_url:
        raise HTTPException(status_code=500, detail="Ngrok tunnel is not active")

    # Determine phone numbers
    phone_number_to_call = config.phone_number_to_call
    twilio_phone_number = config.twilio_phone_number or os.getenv("TWILIO_PHONE_NUMBER") or "+15554443333"

    print(f"DEBUG: Using From Number: {twilio_phone_number}")
    print(f"DEBUG: Using To Number: {phone_number_to_call}")

    # 1. Create Agents
    agents = []
    for a in config.agents:
        if a.voice_id:
            agents.append(Agent(name=a.name, prompt=a.prompt, voice_id=a.voice_id))
        else:
            agents.append(Agent(name=a.name, prompt=a.prompt, voice_id=DEFAULT_VOICE_ID))

    # 2. Create Scenarios
    scenarios = []
    for s in config.scenarios:
        evals = []
        for e in s.evaluations:
            evals.append(Evaluation(name=e.name, prompt=e.prompt))
        scenarios.append(Scenario(name=s.name, prompt=s.prompt, evaluations=evals))

    # 3. Initialize TestRunner (using persistent ngrok url)
    test_runner = TestRunner(
        port=app.state.port,
        ngrok_url=app.state.ngrok_url,
        twilio_phone_number=twilio_phone_number,
        evaluator=LocalEvaluator(),
    )

    # 4. Add Tests
    for scenario in scenarios:
        for agent in agents:
            test = Test(scenario=scenario, agent=agent)
            test_runner.add_test(test)

    # 5. Run Tests
    try:
        raw_results = await test_runner.run_tests(
            phone_number=phone_number_to_call,
            type=TestRunner.OUTBOUND,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")

    # 6. Serialize Results
    serialized_results = []

    if isinstance(raw_results, list):
        for res in raw_results:
            serialized_results.append(make_serializable(res))
    else:
        # Fallback for single result if that ever happens
        serialized_results.append(make_serializable(raw_results))

    overall_passed = all(r.passed for r in serialized_results)

    return TestResponse(results=serialized_results, passed=overall_passed)


if __name__ == "__main__":
    import uvicorn

    # Allow running this file directly for dev convenience
    uvicorn.run(app, host="localhost", port=8000)
