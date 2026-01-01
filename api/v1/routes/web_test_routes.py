"""
Web test routes for web-media-stream scaled testing.

This module provides the /web-test endpoint for running concurrent
WebSocket tests with web-media-stream endpoints.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from models.api import WebScaledTestRequest, WebScaledTestResponse
from services.web_scaled_testing_service import WebScaledTestingService
from telemetrics.logger import logger

router = APIRouter(tags=["Web Test"])

# Store active web test sessions
active_web_tests: dict[str, WebScaledTestingService] = {}


@router.post("/web-test", response_model=WebScaledTestResponse)
async def start_web_test(
    request: WebScaledTestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a scaled WebSocket test with web-media-stream endpoints.

    This endpoint creates multiple parallel WebSocket connections between
    target agent (media-stream endpoint, 8k mulaw) and user agent
    (web-media-stream endpoint, 16k PCM). Uses the exact same logic as test.py.

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

        # Create test service
        test_service = WebScaledTestingService(
            target_agent_uri=request.target_agent_uri,
            user_agent_id=request.user_agent_id,
            ws_url_base=request.ws_url_base,
            recording_path="test_suite_recordings",
        )

        # Generate test ID
        import uuid

        test_id = str(uuid.uuid4())

        # Store active test
        active_web_tests[test_id] = test_service

        # Run test in background
        background_tasks.add_task(_run_web_test_async, test_service, test_id, request)

        logger.info(
            f"Started web test: {test_id}, "
            f"{request.concurrent_requests} concurrent connections, "
            f"timeout: {request.timeout}s"
        )

        return WebScaledTestResponse(
            success=True,
            test_id=test_id,
            message=f"Web test started with {request.concurrent_requests} concurrent connections",
            status={
                "test_id": test_id,
                "concurrent_requests": request.concurrent_requests,
                "timeout": request.timeout,
                "target_agent_uri": request.target_agent_uri,
                "user_agent_id": request.user_agent_id,
                "ws_url_base": request.ws_url_base,
                "status": "running",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting web test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test: {str(e)}")


async def _run_web_test_async(
    test_service: WebScaledTestingService,
    test_id: str,
    request: WebScaledTestRequest,
):
    """Run the web test asynchronously and store results."""
    try:
        result = await test_service.run_concurrent_test(
            concurrent_requests=request.concurrent_requests,
            timeout=request.timeout,
            test_id=test_id,
        )
        logger.info(f"Web test {test_id} completed: {result}")
    except Exception as e:
        logger.error(f"Error running web test {test_id}: {e}")
    finally:
        # Remove from active tests after completion
        if test_id in active_web_tests:
            del active_web_tests[test_id]


@router.get("/web-test/{test_id}")
async def get_web_test_status(test_id: str):
    """
    Get the status of a running or completed web test.

    Args:
        test_id: Unique test identifier

    Returns:
        Test status information
    """
    if test_id not in active_web_tests:
        raise HTTPException(status_code=404, detail=f"Web test '{test_id}' not found")

    return {
        "test_id": test_id,
        "status": "running",
        "message": "Web test is currently running",
    }


@router.get("/web-tests")
async def list_web_tests():
    """List all active web tests."""
    return {
        "total": len(active_web_tests),
        "active_tests": list(active_web_tests.keys()),
    }


@router.delete("/web-test/{test_id}")
async def delete_web_test(test_id: str):
    """
    Delete a web test (stops if running).

    Args:
        test_id: Unique test identifier

    Returns:
        Success confirmation
    """
    if test_id not in active_web_tests:
        raise HTTPException(status_code=404, detail=f"Web test '{test_id}' not found")

    del active_web_tests[test_id]
    return {"success": True, "message": f"Web test '{test_id}' deleted"}
