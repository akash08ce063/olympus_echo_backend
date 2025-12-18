import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import ngrok
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fixa import Agent, Evaluation, Scenario, Test, TestRunner
from fixa.evaluators import LocalEvaluator
from fixa.test_runner.server import active_pairs, call_status
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

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
    transcript: Union[str, List[Dict[str, str]]]
    recording_url: str
    error: Optional[str] = None
    evaluations: Dict[str, Any] = {}


class TestResponse(BaseModel):
    results: List[TestResultModel]
    passed: bool


class JobStartedResponse(BaseModel):
    message: str
    job_id: str


DEFAULT_VOICE_ID = "b7d50908-b17c-442d-ad8d-810c63997ed9"


# Log Streaming Setup
# Note: log_queue will be initialized in the lifespan manager to ensure
# it's created with the correct event loop
log_queue: Optional[asyncio.Queue] = None


def sink(message):
    """
    Loguru sink that puts log records into an async queue.
    The message object from loguru is a string-like object with .record dict.
    """
    global log_queue
    if log_queue is None:
        # Queue not initialized yet, skip logging
        return

    try:
        # Get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a structured event for the log
            event = {"type": "log", "payload": str(message)}
            asyncio.create_task(log_queue.put(event))
        else:
            pass
    except Exception:
        pass


# Configure loguru to use our sink (broadcasts logs to the queue)
logger.add(sink, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# Lifespan Manager for persistent Infrastructure


@asynccontextmanager
async def lifespan(app: FastAPI):
    global log_queue

    # Startup: Initialize log queue with the current event loop
    log_queue = asyncio.Queue()
    app.state.log_queue = log_queue
    print("Log queue initialized")

    # Startup: Initialize ngrok
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if not ngrok_token:
        print("WARNING: NGROK_AUTH_TOKEN not found in environment.")

    # API server runs on 8000, ngrok tunnels to 8765
    api_port = 8000
    ngrok_port = 8765
    listener = None
    print(f"Starting ngrok tunnel on port {ngrok_port}...")
    try:
        # ngrok.forward returns a listener object.
        # Using type ignore as linter sometimes flags await on Listener
        listener = await ngrok.forward(ngrok_port, authtoken=ngrok_token)  # type: ignore
        app.state.ngrok_url = listener.url()
        app.state.ngrok_listener = listener  # Store for cleanup
        app.state.api_port = api_port
        app.state.ngrok_port = ngrok_port
        print(f"Ngrok tunnel established: {app.state.ngrok_url}")
        print(f"API server will run on port {api_port}, ngrok forwards to port {ngrok_port}")
    except Exception as e:
        print(f"Failed to start ngrok: {e}")
        # We don't exit here so the server can still start, strictly speaking,
        # but tests will likely fail if they depend on ngrok.
        app.state.ngrok_url = None
        app.state.ngrok_listener = None

    # Startup: Initialize TestRunner and start its server
    # This server will be shared across all test requests
    if app.state.ngrok_url:
        try:
            print("Initializing shared TestRunner...")
            twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER") or "+15554443333"
            test_runner = TestRunner(
                port=ngrok_port,
                ngrok_url=app.state.ngrok_url,
                twilio_phone_number=twilio_phone_number,
                evaluator=LocalEvaluator(),
            )
            # Start the TestRunner's server once
            await test_runner._start_server()
            app.state.test_runner = test_runner
            print(f"TestRunner initialized and server started on port {ngrok_port}")
        except Exception as e:
            print(f"Failed to initialize TestRunner: {e}")
            app.state.test_runner = None
    else:
        print("Skipping TestRunner initialization (ngrok not available)")
        app.state.test_runner = None

    yield

    # Shutdown: Cleanup resources
    print("Shutting down...")

    # Stop TestRunner server
    if hasattr(app.state, "test_runner") and app.state.test_runner:
        try:
            print("Stopping TestRunner server...")
            await app.state.test_runner._stop_server()
            print("TestRunner server stopped")
        except Exception as e:
            print(f"Error stopping TestRunner server: {e}")

    # Close ngrok tunnel
    if hasattr(app.state, "ngrok_listener") and app.state.ngrok_listener:
        try:
            print("Closing ngrok tunnel...")
            await app.state.ngrok_listener.close()
            print("Ngrok tunnel closed")
        except Exception as e:
            print(f"Error closing ngrok tunnel: {e}")

    # Clear the log queue and cancel any pending tasks
    if log_queue:
        try:
            # Drain the queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            print("Log queue cleared")
        except Exception as e:
            print(f"Error clearing log queue: {e}")

    print("Shutdown complete")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_serializable(res: Any) -> TestResultModel:
    """
    Helper to safely convert a Fixa TestResult object into our Pydantic model.
    Handles potential missing attributes gracefully.
    """
    if hasattr(res, "model_dump"):
        pass

    # Safe extraction
    test_obj = getattr(res, "test", None)

    agent_obj = getattr(res, "agent", None)
    if not agent_obj and test_obj:
        agent_obj = getattr(test_obj, "agent", None)
    agent_name = getattr(agent_obj, "name", "Unknown") if agent_obj else "Unknown"

    scenario_obj = getattr(res, "scenario", None)
    if not scenario_obj and test_obj:
        scenario_obj = getattr(test_obj, "scenario", None)
    scenario_name = getattr(scenario_obj, "name", "Unknown") if scenario_obj else "Unknown"

    transcript = getattr(res, "transcript", [])
    if not transcript:
        transcript = []

    # Map evaluations
    evals_dict = {}
    eval_response = getattr(res, "evaluation_results", None)
    if eval_response:
        # access the inner list of EvaluationResult objects
        eval_results_list = getattr(eval_response, "evaluation_results", [])
        for eval_res in eval_results_list:
            # Each eval_res is an EvaluationResult(name=..., passed=..., reason=...)
            e_name = getattr(eval_res, "name", "Unknown")
            e_passed = getattr(eval_res, "passed", False)
            e_reason = getattr(eval_res, "reason", "")
            evals_dict[e_name] = {"passed": e_passed, "reasoning": e_reason}

    # Check for legacy/direct evaluations attribute if the above is empty
    if not evals_dict:
        evals = getattr(res, "evaluations", {}) or {}
        if isinstance(evals, dict):
            evals_dict = evals

    print(f"DEBUG: Processed evaluations: {evals_dict}")

    # Recording URL - prefer stereo_recording_url
    rec_url = getattr(res, "stereo_recording_url", "")
    if not rec_url:
        rec_url = getattr(res, "recording_url", "") or ""

    return TestResultModel(
        agent=agent_name,
        scenario=scenario_name,
        passed=getattr(res, "passed", False),
        transcript=transcript,
        recording_url=rec_url,
        error=getattr(res, "error", None),
        evaluations=evals_dict,
    )


@app.get("/logs")
async def logs(request: Request):
    """
    SSE endpoint to stream logs and events to the client.
    """
    queue = getattr(app.state, "log_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="Log queue not initialized")

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    # Wait for an event (which is now a dict)
                    event_data = await asyncio.wait_for(queue.get(), timeout=1.0)

                    # sse_starlette automatically serializes 'data' to JSON if it's a dict
                    yield {"data": json.dumps(event_data)}
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            print("SSE connection cancelled during shutdown")
            raise
        finally:
            print("SSE connection closed")

    return EventSourceResponse(event_generator())


async def run_tests_background(config: TestConfig, test_runner: TestRunner, job_id: str):
    """
    Background task to run tests and push results to SSE.
    Uses the shared TestRunner instance.
    """
    global log_queue
    if not log_queue:
        print("CRITICAL: Log queue not active in background task!")
        return

    print(f"Job {job_id}: Starting background test execution...")

    # Push start event
    await log_queue.put({"type": "log", "payload": f"Job {job_id}: Test execution started via background task."})

    try:
        # Clear previous test state
        call_status.clear()
        active_pairs.clear()

        phone_number_to_call = config.phone_number_to_call

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

        # 3. Clear previous tests from the shared TestRunner
        test_runner.tests.clear()

        # 4. Add Tests to the shared TestRunner
        for scenario in scenarios:
            for agent in agents:
                test = Test(scenario=scenario, agent=agent)
                test_runner.add_test(test)

        # 5. Run Tests (using the shared server, no need to start/stop)
        raw_results = await test_runner.run_tests(
            phone_number=phone_number_to_call,
            type=TestRunner.OUTBOUND,
        )

        # print(raw_results)

        # 6. Serialize Results
        serialized_results = []
        if isinstance(raw_results, list):
            for res in raw_results:
                serialized_results.append(make_serializable(res))
        else:
            serialized_results.append(make_serializable(raw_results))

        overall_passed = all(r.passed for r in serialized_results)

        response_payload = TestResponse(results=serialized_results, passed=overall_passed).model_dump()

        # Push Result Event
        print(f"Job {job_id}: Tests finished. Pushing results.")
        await log_queue.put({"type": "result", "payload": response_payload})
        await log_queue.put({"type": "log", "payload": f"Job {job_id}: Results pushed to client."})

    except Exception as e:
        error_msg = f"Error running tests: {str(e)}"
        print(f"Job {job_id} Failed: {error_msg}")
        await log_queue.put({"type": "error", "payload": error_msg})
    finally:
        await log_queue.put({"type": "log", "payload": f"Job {job_id}: Test execution finished."})


@app.post("/test", response_model=JobStartedResponse, status_code=202)
async def run_test(config: TestConfig, background_tasks: BackgroundTasks):
    if not app.state.test_runner:
        raise HTTPException(status_code=500, detail="TestRunner is not initialized")

    job_id = os.urandom(4).hex()  # Simple ID

    # Schedule background task with the shared TestRunner
    background_tasks.add_task(run_tests_background, config=config, test_runner=app.state.test_runner, job_id=job_id)

    return JobStartedResponse(message="Test execution started in background", job_id=job_id)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_graceful_shutdown=5,
    )
