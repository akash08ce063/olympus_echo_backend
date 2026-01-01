"""
User agents CRUD API routes.

This module provides REST API endpoints for managing user agents.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from models.test_suite_models import UserAgentCreate, UserAgentUpdate, UserAgent
from services.user_agent_service import UserAgentService
from telemetrics.logger import logger

router = APIRouter(prefix="/user-agents", tags=["User Agents"])


# Dependency to get user agent service
async def get_user_agent_service() -> UserAgentService:
    """Dependency to get user agent service instance."""
    service = UserAgentService()
    try:
        yield service
    finally:
        await service.close()


class UserAgentListResponse(BaseModel):
    """Response for listing user agents."""
    user_agents: List[UserAgent]
    total: int
    limit: int
    offset: int


@router.post("", response_model=UserAgent)
async def create_user_agent(
    data: UserAgentCreate,
    service: UserAgentService = Depends(get_user_agent_service),
):
    """Create a new user agent."""
    try:
        agent_id = await service.create_user_agent(data.user_id, data)
        agent = await service.get_user_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=500, detail="Failed to retrieve created user agent")
        return agent
    except Exception as e:
        logger.error(f"Error creating user agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create user agent: {str(e)}")


@router.get("/{agent_id}", response_model=UserAgent)
async def get_user_agent(
    agent_id: UUID,
    service: UserAgentService = Depends(get_user_agent_service),
):
    """Get a user agent by ID."""
    agent = await service.get_user_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"User agent '{agent_id}' not found")
    return agent


@router.get("", response_model=UserAgentListResponse)
async def list_user_agents(
    user_id: UUID = Query(..., description="User ID to filter user agents"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: UserAgentService = Depends(get_user_agent_service),
):
    """List user agents for a user."""
    try:
        user_agents = await service.get_user_agents_by_user(user_id, limit, offset)
        total = await service.get_user_agent_count(user_id)
        return UserAgentListResponse(
            user_agents=user_agents,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing user agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list user agents: {str(e)}")


@router.put("/{agent_id}", response_model=UserAgent)
async def update_user_agent(
    agent_id: UUID,
    data: UserAgentUpdate,
    service: UserAgentService = Depends(get_user_agent_service),
):
    """Update a user agent."""
    # Check if agent exists
    existing_agent = await service.get_user_agent(agent_id)
    if not existing_agent:
        raise HTTPException(status_code=404, detail=f"User agent '{agent_id}' not found")

    try:
        success = await service.update_user_agent(agent_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update user agent")

        # Return updated agent
        updated_agent = await service.get_user_agent(agent_id)
        return updated_agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update user agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_user_agent(
    agent_id: UUID,
    service: UserAgentService = Depends(get_user_agent_service),
):
    """Delete a user agent."""
    # Check if agent exists
    existing_agent = await service.get_user_agent(agent_id)
    if not existing_agent:
        raise HTTPException(status_code=404, detail=f"User agent '{agent_id}' not found")

    try:
        success = await service.delete_user_agent(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user agent")

        return {"success": True, "message": f"User agent '{agent_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user agent: {str(e)}")
