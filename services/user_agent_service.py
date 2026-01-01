"""
User agent CRUD service.

This module provides CRUD operations for user agents.
"""

from typing import List, Optional
from uuid import UUID

from services.database_service import DatabaseService
from models.test_suite_models import UserAgentCreate, UserAgentUpdate, UserAgent
from telemetrics.logger import logger


class UserAgentService(DatabaseService[UserAgent]):
    """Service for user agent CRUD operations."""

    def __init__(self):
        super().__init__("user_agents")

    async def create_user_agent(self, user_id: UUID, data: UserAgentCreate) -> UUID:
        """Create a new user agent."""
        agent_data = data.model_dump()
        agent_data["user_id"] = user_id
        return await self.create(agent_data)

    async def get_user_agent(self, agent_id: UUID) -> Optional[UserAgent]:
        """Get a user agent by ID."""
        result = await self.get_by_id(agent_id)
        if result:
            return UserAgent(**result)
        return None

    async def get_user_agents_by_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[UserAgent]:
        """Get user agents for a user."""
        results = await self.get_all_by_user(user_id, limit, offset)
        return [UserAgent(**result) for result in results]

    async def update_user_agent(self, agent_id: UUID, data: UserAgentUpdate) -> bool:
        """Update a user agent."""
        update_data = data.model_dump(exclude_unset=True)
        return await self.update(agent_id, update_data)

    async def delete_user_agent(self, agent_id: UUID) -> bool:
        """Delete a user agent."""
        return await self.delete(agent_id)

    async def get_user_agent_count(self, user_id: UUID) -> int:
        """Get count of user agents for a user."""
        return await self.count_by_user(user_id)
