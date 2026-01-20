"""
Test execution API routes.

This module provides REST API endpoints for running test cases and test suites.
For test runs history, recordings, and transcripts, see test_runs_routes.py
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Request
from pydantic import BaseModel

from services.test_execution_service import TestExecutionService
from services.test_suite_service import TestSuiteService
from services.test_case_service import TestCaseService
from telemetrics.logger import logger

router = APIRouter(prefix="/test-execution", tags=["Test Execution"])


# Dependency to get test execution service
async def get_test_execution_service() -> TestExecutionService:
    """Dependency to get test execution service instance."""
    service = TestExecutionService()
    try:
        yield service
    finally:
        await service.close()


class RunTestSuiteRequest(BaseModel):
    """Request model for running a test suite."""
    concurrent_calls: Optional[int] = Query(1, ge=1, le=10, description="Number of concurrent calls (overrides default)")
    execution_mode: Optional[str] = Query("sequential", description="Execution mode: 'sequential' or 'parallel'")


class RunTestCaseRequest(BaseModel):
    """Request model for running a test case."""
    concurrent_calls: Optional[int] = Query(1, ge=1, le=10, description="Number of concurrent calls (overrides default)")


class TestExecutionResponse(BaseModel):
    """Response model for test execution."""
    success: bool
    test_run_id: UUID
    message: str
    status: dict


@router.post("/run-suite/{suite_id}", response_model=TestExecutionResponse)
async def run_test_suite(
    suite_id: UUID,
    request: RunTestSuiteRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Query(..., description="User ID who is running the test"),
    request_obj: Request = None,
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of all active test cases in a test suite.

    Args:
        suite_id: ID of the test suite to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
        request_obj: FastAPI request object to capture headers
        service: Test execution service

    Returns:
        Test execution response with run ID
    """
    test_suite_service = None
    test_case_service = None
    active_cases = []

    try:
        # Validate that the test suite exists and user has access
        test_suite_service = TestSuiteService()
        test_suite = await test_suite_service.get_test_suite(suite_id)
        if not test_suite:
            raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")

        if test_suite.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to run tests for this test suite"
            )

        # Check if there are active test cases
        test_case_service = TestCaseService()
        active_cases = await test_case_service.get_test_cases_by_suite(suite_id, include_inactive=False)
        if not active_cases:
            raise HTTPException(
                status_code=400,
                detail="No active test cases found in this test suite"
            )

        # Get the request IDs from header (comma-separated for concurrent calls)
        request_ids_header = request_obj.headers.get("x-pranthora-callid")
        if not request_ids_header:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header is required"
            )
        
        # Parse comma-separated request IDs
        request_ids = [rid.strip() for rid in request_ids_header.split(",") if rid.strip()]
        if not request_ids:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header must contain at least one request ID"
            )

        # Start the test execution
        execution_mode = request.execution_mode or "sequential"
        if execution_mode not in ["sequential", "parallel"]:
            raise HTTPException(
                status_code=400,
                detail="execution_mode must be 'sequential' or 'parallel'"
            )
        
        test_run_id = await service.run_test_suite(
            test_suite_id=suite_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls,
            request_ids=request_ids,
            execution_mode=execution_mode
        )

        return TestExecutionResponse(
            success=True,
            test_run_id=test_run_id,
            message=f"Started execution of test suite '{suite_id}' with {len(active_cases)} test cases",
            status={
                "test_run_id": str(test_run_id),
                "suite_id": str(suite_id),
                "status": "running",
                "total_test_cases": len(active_cases),
                "concurrent_calls": request.concurrent_calls,
                "execution_mode": execution_mode
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test suite execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test execution: {str(e)}")
    finally:
        if test_suite_service:
            await test_suite_service.close()
        if test_case_service:
            await test_case_service.close()


@router.post("/run-case/{case_id}", response_model=TestExecutionResponse)
async def run_test_case(
    case_id: UUID,
    request: RunTestCaseRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Query(..., description="User ID who is running the test"),
    request_obj: Request = None,
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of a single test case.

    Args:
        case_id: ID of the test case to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
        request_obj: FastAPI request object to capture headers
        service: Test execution service

    Returns:
        Test execution response with run ID
    """
    try:
        # Validate that the test case exists and user has access
        test_case_service = TestCaseService()
        try:
            test_case = await test_case_service.get_test_case(case_id)
            if not test_case:
                raise HTTPException(status_code=404, detail=f"Test case '{case_id}' not found")

            # Check user access via test suite
            test_suite_service = TestSuiteService()
            try:
                test_suite = await test_suite_service.get_test_suite(test_case.test_suite_id)
                if not test_suite or test_suite.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="You don't have permission to run this test case"
                    )
            finally:
                await test_suite_service.close()

            # Check if test case is active
            if not test_case.is_active:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot run inactive test case"
                )

        finally:
            await test_case_service.close()

        # Get the request IDs from header (comma-separated for concurrent calls)
        request_ids_header = request_obj.headers.get("x-pranthora-callid")
        if not request_ids_header:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header is required"
            )
        
        # Parse comma-separated request IDs
        request_ids = [rid.strip() for rid in request_ids_header.split(",") if rid.strip()]
        if not request_ids:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header must contain at least one request ID"
            )

        # Start the test execution
        test_run_id = await service.run_single_test_case(
            test_case_id=case_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls,
            request_ids=request_ids
        )

        return TestExecutionResponse(
            success=True,
            test_run_id=test_run_id,
            message=f"Started execution of test case '{case_id}'",
            status={
                "test_run_id": str(test_run_id),
                "case_id": str(case_id),
                "status": "running",
                "concurrent_calls": request.concurrent_calls
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test case execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test execution: {str(e)}")


@router.get("/status/{run_id}")
async def get_test_run_status(
    run_id: UUID,
    user_id: UUID = Query(..., description="User ID for authorization"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Get the status of a test run.

    Args:
        run_id: Test run ID
        user_id: User ID for authorization
        service: Test execution service

    Returns:
        Test run status information
    """
    try:
        from services.test_history_service import TestRunHistoryService

        run_service = TestRunHistoryService()
        try:
            test_run = await run_service.get_test_run(run_id)
            if not test_run:
                raise HTTPException(status_code=404, detail=f"Test run '{run_id}' not found")

            # Check user access
            if test_run.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to view this test run"
                )

            # Get test run with results
            test_run_with_results = await run_service.get_test_run_with_results(run_id)

            return {
                "test_run_id": str(run_id),
                "status": test_run.status,
                "started_at": test_run.started_at,
                "completed_at": test_run.completed_at,
                "total_test_cases": test_run.total_test_cases,
                "passed_count": test_run.passed_count,
                "failed_count": test_run.failed_count,
                "alert_count": test_run.alert_count,
                "test_case_results": [
                    {
                        "id": str(result.id),
                        "test_case_id": str(result.test_case_id),
                        "status": result.status,
                        "started_at": result.started_at,
                        "completed_at": result.completed_at,
                        "error_message": result.error_message
                    }
                    for result in (test_run_with_results.test_case_results if test_run_with_results else [])
                ]
            }

        finally:
            await run_service.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test run status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test run status: {str(e)}")
