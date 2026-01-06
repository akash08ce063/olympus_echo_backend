"""
Target agent CRUD service.

This module provides CRUD operations for target agents.
"""

from typing import List, Optional
from uuid import UUID

from services.database_service import DatabaseService
from services.test_suite_service import TestSuiteService
from models.test_suite_models import TargetAgentCreate, TargetAgentUpdate, TargetAgent
from telemetrics.logger import logger


class TargetAgentService(DatabaseService[TargetAgent]):
    """Service for target agent CRUD operations."""

    def __init__(self):
        super().__init__("target_agents")

    async def create_target_agent(self, user_id: UUID, data: TargetAgentCreate) -> UUID:
        """Create a new target agent."""
        agent_data = data.model_dump()
        agent_data["user_id"] = user_id
        return await self.create(agent_data)

    async def get_target_agent(self, agent_id: UUID) -> Optional[TargetAgent]:
        """Get a target agent by ID."""
        result = await self.get_by_id(agent_id)
        if result:
            return TargetAgent(**result)
        return None

    async def get_target_agents_by_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[TargetAgent]:
        """Get target agents for a user."""
        results = await self.get_all_by_user(user_id, limit, offset)
        return [TargetAgent(**result) for result in results]

    async def update_target_agent(self, agent_id: UUID, data: TargetAgentUpdate) -> bool:
        """Update a target agent."""
        update_data = data.model_dump(exclude_unset=True)
        return await self.update(agent_id, update_data)

    async def delete_target_agent(self, agent_id: UUID) -> bool:
        """Delete a target agent (nullifies associated test suite references first)."""
        # Replace target_agent_id with null in test suites (don't delete them)
        test_suite_service = TestSuiteService()
        updated_count = await test_suite_service.nullify_target_agent_references(agent_id)
        if updated_count > 0:
            logger.info(f"Set target_agent_id to null for {updated_count} test suites associated with target agent {agent_id}")

        # Delete from local database
        return await self.delete(agent_id)

    async def get_target_agent_count(self, user_id: UUID) -> int:
        """Get count of target agents for a user."""
        return await self.count_by_user(user_id)
