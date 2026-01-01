"""
Twilio test routes for media-stream scaled testing.

This module provides the /twilio-test endpoint for running concurrent
WebSocket tests with Twilio media-stream endpoints.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from models.api import ScaledTestRequest, ScaledTestResponse
from services.scaled_testing_service import ScaledTestingService
from telemetrics.logger import logger

router = APIRouter(tags=["Twilio Test"])

# Store active Twilio test sessions
active_twilio_tests: dict[str, ScaledTestingService] = {}


@router.post("/twilio-test", response_model=ScaledTestResponse)
async def start_twilio_test(
    request: ScaledTestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a scaled WebSocket test with Twilio media-stream endpoints.

    This endpoint creates multiple parallel WebSocket connections between
    target and user agents using Twilio media-stream protocol (8kHz μ-law).

    Args:
        request: Test configuration including agent URIs, concurrency, timeout, etc.
        background_tasks: FastAPI background tasks for async execution

    Returns:
        Response with test ID and initial status
    """
    try:
        # Validate inputs
        if request.concurrent_requests < 1:
            raise HTTPException(
                status_code=400, detail="concurrent_requests must be at least 1"
            )
        if request.timeout < 1:
            raise HTTPException(
                status_code=400, detail="timeout must be at least 1 second"
            )
        if request.sample_rate < 1000 or request.sample_rate > 48000:
            raise HTTPException(
                status_code=400,
                detail="sample_rate must be between 1000 and 48000 Hz",
            )

        valid_encodings = ["mulaw", "pcm16", "pcm"]
        if request.encoding.lower() not in valid_encodings:
            raise HTTPException(
                status_code=400,
                detail=f"encoding must be one of: {', '.join(valid_encodings)}",
            )

        # Create test service
        test_service = ScaledTestingService(
            target_agent_uri=request.target_agent_uri,
            user_agent_uri=request.user_agent_uri,
            sample_rate=request.sample_rate,
            encoding=request.encoding,
            recording_path="test_suite_recordings",
        )

        # Generate test ID
        import uuid

        test_id = str(uuid.uuid4())

        # Store active test
        active_twilio_tests[test_id] = test_service

        # Run test in background
        background_tasks.add_task(_run_twilio_test_async, test_service, test_id, request)

        logger.info(
            f"Started Twilio test: {test_id}, "
            f"{request.concurrent_requests} concurrent connections, "
            f"timeout: {request.timeout}s"
        )

        return ScaledTestResponse(
            success=True,
            test_id=test_id,
            message=f"Twilio test started with {request.concurrent_requests} concurrent connections",
            status={
                "test_id": test_id,
                "concurrent_requests": request.concurrent_requests,
                "timeout": request.timeout,
                "target_agent_uri": request.target_agent_uri,
                "user_agent_uri": request.user_agent_uri,
                "sample_rate": request.sample_rate,
                "encoding": request.encoding,
                "status": "running",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting Twilio test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test: {str(e)}")


async def _run_twilio_test_async(
    test_service: ScaledTestingService,
    test_id: str,
    request: ScaledTestRequest,
):
    """Run the Twilio test asynchronously and store results."""
    try:
        result = await test_service.run_concurrent_test(
            concurrent_requests=request.concurrent_requests,
            timeout=request.timeout,
            test_id=test_id,
        )
        logger.info(f"Twilio test {test_id} completed: {result}")
    except Exception as e:
        logger.error(f"Error running Twilio test {test_id}: {e}")
    finally:
        # Remove from active tests after completion
        if test_id in active_twilio_tests:
            del active_twilio_tests[test_id]


@router.get("/twilio-test/{test_id}")
async def get_twilio_test_status(test_id: str):
    """
    Get the status of a running or completed Twilio test.

    Args:
        test_id: Unique test identifier

    Returns:
        Test status information
    """
    if test_id not in active_twilio_tests:
        raise HTTPException(status_code=404, detail=f"Twilio test '{test_id}' not found")

    return {
        "test_id": test_id,
        "status": "running",
        "message": "Twilio test is currently running",
    }


@router.get("/twilio-tests")
async def list_twilio_tests():
    """List all active Twilio tests."""
    return {
        "total": len(active_twilio_tests),
        "active_tests": list(active_twilio_tests.keys()),
    }


@router.delete("/twilio-test/{test_id}")
async def delete_twilio_test(test_id: str):
    """
    Delete a Twilio test (stops if running).

    Args:
        test_id: Unique test identifier

    Returns:
        Success confirmation
    """
    if test_id not in active_twilio_tests:
        raise HTTPException(status_code=404, detail=f"Twilio test '{test_id}' not found")

    del active_twilio_tests[test_id]
    return {"success": True, "message": f"Twilio test '{test_id}' deleted"}
