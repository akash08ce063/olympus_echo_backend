"""
Test cases CRUD API routes.

This module provides REST API endpoints for managing test cases.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from models.test_suite_models import TestCaseCreate, TestCaseUpdate, TestCase
from services.test_case_service import TestCaseService
from telemetrics.logger import logger

router = APIRouter(prefix="/test-cases", tags=["Test Cases"])


# Dependency to get test case service
async def get_test_case_service() -> TestCaseService:
    """Dependency to get test case service instance."""
    service = TestCaseService()
    try:
        yield service
    finally:
        await service.close()


class TestCaseListResponse(BaseModel):
    """Response for listing test cases."""
    test_cases: List[TestCase]
    total: int
    limit: int
    offset: int


class TestCaseReorderRequest(BaseModel):
    """Request for reordering test cases."""
    case_orders: List[dict] = [
        {
            "case_id": "UUID of test case",
            "order_index": "New order index (integer)"
        }
    ]


@router.post("", response_model=TestCase)
async def create_test_case(
    data: TestCaseCreate,
    service: TestCaseService = Depends(get_test_case_service),
):
    """Create a new test case."""
    try:
        case_id = await service.create_test_case(data)
        case = await service.get_test_case(case_id)
        if not case:
            raise HTTPException(status_code=500, detail="Failed to retrieve created test case")
        return case
    except Exception as e:
        logger.error(f"Error creating test case: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test case: {str(e)}")


@router.get("/{case_id}", response_model=TestCase)
async def get_test_case(
    case_id: UUID,
    service: TestCaseService = Depends(get_test_case_service),
):
    """Get a test case by ID."""
    case = await service.get_test_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Test case '{case_id}' not found")
    return case


@router.get("", response_model=TestCaseListResponse)
async def list_test_cases(
    suite_id: UUID = Query(..., description="Test suite ID to filter test cases"),
    include_inactive: bool = Query(False, description="Include inactive test cases"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TestCaseService = Depends(get_test_case_service),
):
    """List test cases for a test suite."""
    try:
        test_cases = await service.get_test_cases_by_suite(
            suite_id, include_inactive, limit, offset
        )
        total = await service.get_test_case_count(suite_id, include_inactive)
        return TestCaseListResponse(
            test_cases=test_cases,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing test cases: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test cases: {str(e)}")


@router.put("/{case_id}", response_model=TestCase)
async def update_test_case(
    case_id: UUID,
    data: TestCaseUpdate,
    service: TestCaseService = Depends(get_test_case_service),
):
    """Update a test case."""
    # Check if case exists
    existing_case = await service.get_test_case(case_id)
    if not existing_case:
        raise HTTPException(status_code=404, detail=f"Test case '{case_id}' not found")

    try:
        success = await service.update_test_case(case_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update test case")

        # Return updated case
        updated_case = await service.get_test_case(case_id)
        return updated_case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating test case {case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update test case: {str(e)}")


@router.put("/reorder/{suite_id}")
async def reorder_test_cases(
    suite_id: UUID,
    request: TestCaseReorderRequest,
    service: TestCaseService = Depends(get_test_case_service),
):
    """Reorder test cases within a test suite."""
    try:
        success = await service.reorder_test_cases(suite_id, request.case_orders)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to reorder test cases")

        return {"success": True, "message": f"Test cases reordered for suite '{suite_id}'"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reordering test cases for suite {suite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reorder test cases: {str(e)}")


@router.delete("/{case_id}")
async def delete_test_case(
    case_id: UUID,
    service: TestCaseService = Depends(get_test_case_service),
):
    """Delete a test case."""
    # Check if case exists
    existing_case = await service.get_test_case(case_id)
    if not existing_case:
        raise HTTPException(status_code=404, detail=f"Test case '{case_id}' not found")

    try:
        success = await service.delete_test_case(case_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete test case")

        return {"success": True, "message": f"Test case '{case_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test case {case_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete test case: {str(e)}")
