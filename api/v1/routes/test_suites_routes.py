"""
Test suites CRUD API routes.

This module provides REST API endpoints for managing test suites.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from models.test_suite_models import (
    TestSuiteCreate, TestSuiteCreateRequest, TestSuiteUpdate, TestSuite, TestSuiteWithRelations
)
from services.test_suite_service import TestSuiteService
from telemetrics.logger import logger

router = APIRouter(prefix="/test-suites", tags=["Test Suites"])


# Dependency to get test suite service
async def get_test_suite_service() -> TestSuiteService:
    """Dependency to get test suite service instance."""
    service = TestSuiteService()
    try:
        yield service
    finally:
        await service.close()


class TestSuiteListResponse(BaseModel):
    """Response for listing test suites."""
    test_suites: List[TestSuite]
    total: int
    limit: int
    offset: int


@router.post("", response_model=TestSuite)
async def create_test_suite(
    data: TestSuiteCreateRequest,
    user_id: Optional[UUID] = Query(None, description="User ID (can also be in request body)"),
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """Create a new test suite.

    User ID can be provided either in the request body or as a query parameter.
    Allows creating test suites with just name and description (target_agent_id and user_agent_id are optional).
    """
    try:
        # Use user_id from query param if provided, otherwise from body
        actual_user_id = user_id if user_id is not None else data.user_id

        if actual_user_id is None:
            raise HTTPException(status_code=400, detail="user_id is required (in body or as query parameter)")

        # Create the full TestSuiteCreate object
        suite_data = TestSuiteCreate(
            user_id=actual_user_id,
            name=data.name,
            description=data.description,
            target_agent_id=data.target_agent_id,
            user_agent_id=data.user_agent_id
        )

        suite_id = await service.create_test_suite(actual_user_id, suite_data)
        suite = await service.get_test_suite(suite_id)
        if not suite:
            raise HTTPException(status_code=500, detail="Failed to retrieve created test suite")
        return suite
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating test suite: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create test suite: {str(e)}")


@router.get("/{suite_id}", response_model=TestSuite)
async def get_test_suite(
    suite_id: UUID,
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """Get a test suite by ID."""
    suite = await service.get_test_suite(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")
    return suite


@router.get("/{suite_id}/details", response_model=TestSuiteWithRelations)
async def get_test_suite_with_relations(
    suite_id: UUID,
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """Get a test suite with all related entities (agents, test cases)."""
    suite = await service.get_test_suite_with_relations(suite_id)
    if not suite:
        raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")
    return suite


@router.get("", response_model=TestSuiteListResponse)
async def list_test_suites(
    user_id: UUID = Query(..., description="User ID to filter test suites"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """List test suites for a user."""
    try:
        test_suites = await service.get_test_suites_by_user(user_id, limit, offset)
        total = await service.get_test_suite_count(user_id)
        return TestSuiteListResponse(
            test_suites=test_suites,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing test suites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test suites: {str(e)}")


@router.put("/{suite_id}", response_model=TestSuite)
async def update_test_suite(
    suite_id: UUID,
    data: TestSuiteUpdate,
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """Update a test suite."""
    # Check if suite exists
    existing_suite = await service.get_test_suite(suite_id)
    if not existing_suite:
        raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")

    try:
        success = await service.update_test_suite(suite_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update test suite")

        # Return updated suite
        updated_suite = await service.get_test_suite(suite_id)
        return updated_suite
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating test suite {suite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update test suite: {str(e)}")


@router.delete("/{suite_id}")
async def delete_test_suite(
    suite_id: UUID,
    service: TestSuiteService = Depends(get_test_suite_service),
):
    """Delete a test suite."""
    # Check if suite exists
    existing_suite = await service.get_test_suite(suite_id)
    if not existing_suite:
        raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")

    try:
        success = await service.delete_test_suite(suite_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete test suite")

        return {"success": True, "message": f"Test suite '{suite_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting test suite {suite_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete test suite: {str(e)}")
