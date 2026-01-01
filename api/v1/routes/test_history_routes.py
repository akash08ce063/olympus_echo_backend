"""
Test history read-only API routes.

This module provides REST API endpoints for viewing test run history and results.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from models.test_suite_models import (
    TestRunHistory, TestCaseResult, TestAlert,
    TestRunWithResults, TestCaseResultWithAlerts
)
from services.test_history_service import (
    TestRunHistoryService, TestCaseResultService, TestAlertService
)
from telemetrics.logger import logger

router = APIRouter(prefix="/test-history", tags=["Test History"])


# Dependencies
async def get_test_run_service() -> TestRunHistoryService:
    """Dependency to get test run history service instance."""
    service = TestRunHistoryService()
    try:
        yield service
    finally:
        await service.close()


async def get_test_result_service() -> TestCaseResultService:
    """Dependency to get test case result service instance."""
    service = TestCaseResultService()
    try:
        yield service
    finally:
        await service.close()


async def get_test_alert_service() -> TestAlertService:
    """Dependency to get test alert service instance."""
    service = TestAlertService()
    try:
        yield service
    finally:
        await service.close()


# Response models
class TestRunListResponse(BaseModel):
    """Response for listing test runs."""
    test_runs: List[TestRunHistory]
    total: int
    limit: int
    offset: int


class TestResultListResponse(BaseModel):
    """Response for listing test case results."""
    test_results: List[TestCaseResult]
    total: int
    limit: int
    offset: int


class TestAlertListResponse(BaseModel):
    """Response for listing test alerts."""
    test_alerts: List[TestAlert]
    total: int
    limit: int
    offset: int


# Test Run History endpoints
@router.get("/runs/{run_id}", response_model=TestRunHistory)
async def get_test_run(
    run_id: UUID,
    service: TestRunHistoryService = Depends(get_test_run_service),
):
    """Get a test run by ID."""
    test_run = await service.get_test_run(run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run '{run_id}' not found")
    return test_run


@router.get("/runs/{run_id}/details", response_model=TestRunWithResults)
async def get_test_run_with_results(
    run_id: UUID,
    service: TestRunHistoryService = Depends(get_test_run_service),
):
    """Get a test run with all its results."""
    test_run = await service.get_test_run_with_results(run_id)
    if not test_run:
        raise HTTPException(status_code=404, detail=f"Test run '{run_id}' not found")
    return test_run


@router.get("/runs", response_model=TestRunListResponse)
async def list_test_runs(
    suite_id: Optional[UUID] = Query(None, description="Test suite ID to filter runs"),
    user_id: Optional[UUID] = Query(None, description="User ID to filter runs"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TestRunHistoryService = Depends(get_test_run_service),
):
    """List test runs (by suite or user)."""
    if not suite_id and not user_id:
        raise HTTPException(
            status_code=400,
            detail="Either suite_id or user_id must be provided"
        )

    try:
        if suite_id:
            test_runs = await service.get_test_runs_by_suite(suite_id, limit, offset)
            # For simplicity, we'll count total runs for the suite
            total = len(test_runs) + offset  # Approximate total
        else:
            test_runs = await service.get_test_runs_by_user(user_id, limit, offset)
            total = len(test_runs) + offset  # Approximate total

        return TestRunListResponse(
            test_runs=test_runs,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing test runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test runs: {str(e)}")


# Test Case Results endpoints
@router.get("/results/{result_id}", response_model=TestCaseResult)
async def get_test_result(
    result_id: UUID,
    service: TestCaseResultService = Depends(get_test_result_service),
):
    """Get a test case result by ID."""
    result = await service.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Test result '{result_id}' not found")
    return result


@router.get("/results/{result_id}/details", response_model=TestCaseResultWithAlerts)
async def get_test_result_with_alerts(
    result_id: UUID,
    service: TestCaseResultService = Depends(get_test_result_service),
):
    """Get a test case result with all its alerts."""
    result = await service.get_result_with_alerts(result_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Test result '{result_id}' not found")
    return result


@router.get("/results", response_model=TestResultListResponse)
async def list_test_results(
    run_id: Optional[UUID] = Query(None, description="Test run ID to filter results"),
    case_id: Optional[UUID] = Query(None, description="Test case ID to filter results"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TestCaseResultService = Depends(get_test_result_service),
):
    """List test case results (by run or case)."""
    if not run_id and not case_id:
        raise HTTPException(
            status_code=400,
            detail="Either run_id or case_id must be provided"
        )

    try:
        if run_id:
            test_results = await service.get_results_by_run(run_id)
        else:
            test_results = await service.get_results_by_case(case_id, limit)

        return TestResultListResponse(
            test_results=test_results,
            total=len(test_results),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing test results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test results: {str(e)}")


# Test Alerts endpoints
@router.get("/alerts/{alert_id}", response_model=TestAlert)
async def get_test_alert(
    alert_id: UUID,
    service: TestAlertService = Depends(get_test_alert_service),
):
    """Get a test alert by ID."""
    alert = await service.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Test alert '{alert_id}' not found")
    return alert


@router.get("/alerts", response_model=TestAlertListResponse)
async def list_test_alerts(
    result_id: Optional[UUID] = Query(None, description="Test result ID to filter alerts"),
    severity: Optional[str] = Query(None, description="Alert severity to filter (low, medium, high)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TestAlertService = Depends(get_test_alert_service),
):
    """List test alerts (by result or severity)."""
    if not result_id and not severity:
        raise HTTPException(
            status_code=400,
            detail="Either result_id or severity must be provided"
        )

    try:
        if result_id:
            test_alerts = await service.get_alerts_by_result(result_id)
        else:
            test_alerts = await service.get_alerts_by_severity(severity, limit)

        return TestAlertListResponse(
            test_alerts=test_alerts,
            total=len(test_alerts),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing test alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test alerts: {str(e)}")
