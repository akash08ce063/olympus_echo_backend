"""
Test execution API routes.

This module provides REST API endpoints for running test cases and test suites.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
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
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of all active test cases in a test suite.

    Args:
        suite_id: ID of the test suite to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
        service: Test execution service

    Returns:
        Test execution response with run ID
    """
    try:
        # Validate that the test suite exists and user has access
        test_suite_service = TestSuiteService()
        try:
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

        finally:
            await test_suite_service.close()
            await test_case_service.close()

        # Start the test execution
        test_run_id = await service.run_test_suite(
            suite_id=suite_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls
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
                "concurrent_calls": request.concurrent_calls
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test suite execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test execution: {str(e)}")


@router.post("/run-case/{case_id}", response_model=TestExecutionResponse)
async def run_test_case(
    case_id: UUID,
    request: RunTestCaseRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Query(..., description="User ID who is running the test"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of a single test case.

    Args:
        case_id: ID of the test case to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
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

        # Start the test execution
        test_run_id = await service.run_single_test_case(
            test_case_id=case_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls
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
                        "recording_file_id": str(result.recording_file_id) if result.recording_file_id else None,
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


@router.get("/runs")
async def list_test_runs(
    user_id: UUID = Query(..., description="User ID"),
    suite_id: Optional[UUID] = Query(None, description="Filter by test suite ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    List test runs for a user.

    Args:
        user_id: User ID
        suite_id: Optional test suite ID to filter by
        limit: Maximum number of results
        offset: Number of results to skip
        service: Test execution service

    Returns:
        List of test runs
    """
    try:
        from services.test_history_service import TestRunHistoryService

        run_service = TestRunHistoryService()
        try:
            if suite_id:
                # Filter by suite
                runs = await run_service.get_test_runs_by_suite(suite_id, limit, offset)
                # Filter by user access
                runs = [run for run in runs if run.user_id == user_id]
            else:
                # Get all runs for user
                runs = await run_service.get_test_runs_by_user(user_id, limit, offset)

            return {
                "total": len(runs),
                "runs": [
                    {
                        "id": str(run.id),
                        "test_suite_id": str(run.test_suite_id),
                        "status": run.status,
                        "started_at": run.started_at,
                        "completed_at": run.completed_at,
                        "total_test_cases": run.total_test_cases,
                        "passed_count": run.passed_count,
                        "failed_count": run.failed_count,
                        "alert_count": run.alert_count
                    }
                    for run in runs
                ]
            }

        finally:
            await run_service.close()

    except Exception as e:
        logger.error(f"Error listing test runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test runs: {str(e)}")
