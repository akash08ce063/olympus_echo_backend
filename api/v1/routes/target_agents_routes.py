"""
Target agents CRUD API routes.

This module provides REST API endpoints for managing target agents.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from models.test_suite_models import TargetAgentCreate, TargetAgentUpdate, TargetAgent
from services.target_agent_service import TargetAgentService
from telemetrics.logger import logger

router = APIRouter(prefix="/target-agents", tags=["Target Agents"])


# Dependency to get target agent service
async def get_target_agent_service() -> TargetAgentService:
    """Dependency to get target agent service instance."""
    service = TargetAgentService()
    try:
        yield service
    finally:
        await service.close()


class TargetAgentListResponse(BaseModel):
    """Response for listing target agents."""
    target_agents: List[TargetAgent]
    total: int
    limit: int
    offset: int


@router.post("", response_model=TargetAgent)
async def create_target_agent(
    data: TargetAgentCreate,
    service: TargetAgentService = Depends(get_target_agent_service),
):
    """Create a new target agent."""
    try:
        agent_id = await service.create_target_agent(data.user_id, data)
        agent = await service.get_target_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=500, detail="Failed to retrieve created target agent")
        return agent
    except Exception as e:
        logger.error(f"Error creating target agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create target agent: {str(e)}")


@router.get("/{agent_id}", response_model=TargetAgent)
async def get_target_agent(
    agent_id: UUID,
    service: TargetAgentService = Depends(get_target_agent_service),
):
    """Get a target agent by ID."""
    agent = await service.get_target_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Target agent '{agent_id}' not found")
    return agent


@router.get("", response_model=TargetAgentListResponse)
async def list_target_agents(
    user_id: UUID = Query(..., description="User ID to filter target agents"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    service: TargetAgentService = Depends(get_target_agent_service),
):
    """List target agents for a user."""
    try:
        target_agents = await service.get_target_agents_by_user(user_id, limit, offset)
        total = await service.get_target_agent_count(user_id)
        return TargetAgentListResponse(
            target_agents=target_agents,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing target agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list target agents: {str(e)}")


@router.put("/{agent_id}", response_model=TargetAgent)
async def update_target_agent(
    agent_id: UUID,
    data: TargetAgentUpdate,
    service: TargetAgentService = Depends(get_target_agent_service),
):
    """Update a target agent."""
    # Check if agent exists
    existing_agent = await service.get_target_agent(agent_id)
    if not existing_agent:
        raise HTTPException(status_code=404, detail=f"Target agent '{agent_id}' not found")

    try:
        success = await service.update_target_agent(agent_id, data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update target agent")

        # Return updated agent
        updated_agent = await service.get_target_agent(agent_id)
        return updated_agent
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating target agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update target agent: {str(e)}")


@router.delete("/{agent_id}")
async def delete_target_agent(
    agent_id: UUID,
    service: TargetAgentService = Depends(get_target_agent_service),
):
    """Delete a target agent."""
    # Check if agent exists
    existing_agent = await service.get_target_agent(agent_id)
    if not existing_agent:
        raise HTTPException(status_code=404, detail=f"Target agent '{agent_id}' not found")

    try:
        success = await service.delete_target_agent(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete target agent")

        return {"success": True, "message": f"Target agent '{agent_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting target agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete target agent: {str(e)}")
