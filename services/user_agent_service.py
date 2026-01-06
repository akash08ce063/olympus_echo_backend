"""
User agent CRUD service with Pranthora API integration.

This module provides CRUD operations for user agents, integrating with both
local database and Pranthora backend API.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from services.database_service import DatabaseService
from services.pranthora_api_client import PranthoraApiClient
from services.test_suite_service import TestSuiteService
from models.test_suite_models import UserAgentCreate, UserAgentUpdate, UserAgent
from telemetrics.logger import logger


class UserAgentService(DatabaseService[UserAgent]):
    """Service for user agent CRUD operations with Pranthora API integration."""

    def __init__(self):
        super().__init__("user_agents")

    async def _get_pranthora_client(self) -> PranthoraApiClient:
        """Get Pranthora API client instance."""
        return PranthoraApiClient()

    async def create_user_agent(self, user_id: UUID, data: UserAgentCreate) -> UUID:
        """Create a new user agent in both local DB and Pranthora."""
        # Only include fields that exist in the database
        agent_data = {
            "user_id": str(user_id),
            "name": data.name,
            "system_prompt": data.system_prompt,
            "temperature": data.temperature,
            "pranthora_agent_id": None  # Will be set if Pranthora succeeds
        }

        # First create agent in Pranthora
        async with PranthoraApiClient() as client:
            try:
                pranthora_response = await client.create_agent({
                    "name": data.name,
                    "description": f"User agent: {data.name}",
                    "is_active": True,
                    "system_prompt": data.system_prompt,
                    "temperature": data.temperature
                })

                # Store Pranthora agent ID in our database
                agent_data["pranthora_agent_id"] = pranthora_response["agent"]["id"]
                logger.info(f"Created agent in Pranthora: {agent_data['pranthora_agent_id']}")

            except Exception as e:
                logger.error(f"Failed to create agent in Pranthora: {e}")
                # For now, allow creation without Pranthora agent
                # In production, you might want to raise an error
                pass  # Don't set pranthora_agent_id if creation failed

        # Create agent in our local database
        local_agent_id = await self.create(agent_data)
        logger.info(f"Created agent in local DB: {local_agent_id}")

        return local_agent_id

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
        """Update a user agent in both local DB and Pranthora."""
        update_data = data.model_dump(exclude_unset=True)

        # Update local database first
        local_success = await self.update(agent_id, update_data)
        if not local_success:
            return False

        # Update in Pranthora if we have a pranthora_agent_id
        try:
            # Get the agent to check if it has a pranthora_agent_id
            agent_result = await self.get_by_id(agent_id)
            if agent_result and agent_result.get("pranthora_agent_id"):
                # Prepare data for Pranthora API
                pranthora_update_data = {
                    "name": update_data.get("name"),
                    "system_prompt": update_data.get("system_prompt"),
                    "temperature": update_data.get("temperature")
                }
                # Remove None values
                pranthora_update_data = {k: v for k, v in pranthora_update_data.items() if v is not None}

                async with PranthoraApiClient() as client:
                    await client.update_agent(
                        agent_result["pranthora_agent_id"],
                        pranthora_update_data
                    )
                logger.info(f"Updated agent in Pranthora: {agent_result['pranthora_agent_id']}")
        except Exception as e:
            logger.error(f"Failed to update agent in Pranthora: {e}")
            # Don't fail the local update if Pranthora update fails

        return True

    async def delete_user_agent(self, agent_id: UUID) -> bool:
        """Delete a user agent from both local DB and Pranthora (nullifies associated test suite references)."""
        # Get the agent first to check if it has a pranthora_agent_id
        agent_result = await self.get_by_id(agent_id)
        if not agent_result:
            return False

        # Replace user_agent_id with null in test suites (don't delete them)
        test_suite_service = TestSuiteService()
        updated_count = await test_suite_service.nullify_user_agent_references(agent_id)
        if updated_count > 0:
            logger.info(f"Set user_agent_id to null for {updated_count} test suites associated with user agent {agent_id}")

        # Delete from Pranthora if we have a pranthora_agent_id
        if agent_result.get("pranthora_agent_id"):
            try:
                async with PranthoraApiClient() as client:
                    await client.delete_agent(agent_result["pranthora_agent_id"])
                    logger.info(f"Deleted agent from Pranthora: {agent_result['pranthora_agent_id']}")
            except Exception as e:
                if "not found" in str(e).lower():
                    logger.info(f"Agent {agent_result['pranthora_agent_id']} not found in Pranthora, skipping deletion")
                else:
                    logger.error(f"Failed to delete agent from Pranthora: {e}")
                    # Don't fail the local delete if Pranthora delete fails

        # Delete from local database
        return await self.delete(agent_id)

    async def get_user_agent_count(self, user_id: UUID) -> int:
        """Get count of user agents for a user."""
        return await self.count_by_user(user_id)
