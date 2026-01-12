"""
Test suite routes for scaled WebSocket testing.

This module provides endpoints for running concurrent WebSocket tests
with audio conversion support.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from models.api import ScaledTestRequest, ScaledTestResponse, WebScaledTestRequest, WebScaledTestResponse
from services.scaled_testing_service import ScaledTestingService
from services.web_scaled_testing_service import WebScaledTestingService
from telemetrics.logger import logger

router = APIRouter(tags=["Scaled Tests"])

# Store active test sessions
active_tests: dict[str, ScaledTestingService] = {}
active_web_tests: dict[str, WebScaledTestingService] = {}


@router.post("/twilio-test", response_model=ScaledTestResponse)
async def start_scaled_test(
    request: ScaledTestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a scaled WebSocket test with concurrent connections.

    This endpoint creates multiple parallel WebSocket connections between
    target and user agents, with audio conversion support based on the
    specified encoding and sample rate.

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
        active_tests[test_id] = test_service

        # Run test in background
        background_tasks.add_task(_run_test_async, test_service, test_id, request)

        logger.info(
            f"Started scaled test: {test_id}, "
            f"{request.concurrent_requests} concurrent connections, "
            f"timeout: {request.timeout}s"
        )

        return ScaledTestResponse(
            success=True,
            test_id=test_id,
            message=f"Scaled test started with {request.concurrent_requests} concurrent connections",
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
        logger.error(f"Error starting scaled test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test: {str(e)}")


async def _run_test_async(
    test_service: ScaledTestingService,
    test_id: str,
    request: ScaledTestRequest,
):
    """Run the test asynchronously and store results."""
    try:
        result = await test_service.run_concurrent_test(
            concurrent_requests=request.concurrent_requests,
            timeout=request.timeout,
            test_id=test_id,
        )
        logger.info(f"Test {test_id} completed: {result}")
    except Exception as e:
        logger.error(f"Error running test {test_id}: {e}")
    finally:
        # Remove from active tests after completion
        if test_id in active_tests:
            del active_tests[test_id]


@router.get("/twilio-test/{test_id}")
async def get_test_status(test_id: str):
    """
    Get the status of a running or completed test.

    Args:
        test_id: Unique test identifier

    Returns:
        Test status information
    """
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found")

    return {
        "test_id": test_id,
        "status": "running",
        "message": "Test is currently running",
    }


@router.get("/twilio-tests")
async def list_tests():
    """List all active twilio tests."""
    return {
        "total": len(active_tests),
        "active_tests": list(active_tests.keys()),
    }


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


@router.delete("/twilio-test/{test_id}")
async def delete_test(test_id: str):
    """
    Delete a twilio test (stops if running).

    Args:
        test_id: Unique test identifier

    Returns:
        Success confirmation
    """
    if test_id not in active_tests:
        raise HTTPException(status_code=404, detail=f"Test '{test_id}' not found")

    del active_tests[test_id]
    return {"success": True, "message": f"Test '{test_id}' deleted"}


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


@router.post("/web-test", response_model=WebScaledTestResponse)
async def start_web_scaled_test(
    request: WebScaledTestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a scaled WebSocket test with concurrent connections.

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
            f"Started web scaled test: {test_id}, "
            f"{request.concurrent_requests} concurrent connections, "
            f"timeout: {request.timeout}s"
        )

        return WebScaledTestResponse(
            success=True,
            test_id=test_id,
            message=f"Web scaled test started with {request.concurrent_requests} concurrent connections",
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
        logger.error(f"Error starting web scaled test: {e}")
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

