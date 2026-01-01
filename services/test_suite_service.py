"""
Test suite CRUD service.

This module provides CRUD operations for test suites and related entities.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from services.database_service import DatabaseService
from models.test_suite_models import (
    TestSuiteCreate, TestSuiteUpdate, TestSuite, TestSuiteWithRelations,
    TestCase, TargetAgent, UserAgent
)
from telemetrics.logger import logger


class TestSuiteService(DatabaseService[TestSuite]):
    """Service for test suite CRUD operations."""

    def __init__(self):
        super().__init__("test_suites")

    async def create_test_suite(self, user_id: UUID, data: TestSuiteCreate) -> UUID:
        """Create a new test suite."""
        suite_data = data.model_dump()
        suite_data["user_id"] = user_id
        return await self.create(suite_data)

    async def get_test_suite(self, suite_id: UUID) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        result = await self.get_by_id(suite_id)
        if result:
            return TestSuite(**result)
        return None

    async def get_test_suite_with_relations(self, suite_id: UUID) -> Optional[TestSuiteWithRelations]:
        """Get a test suite with all related entities."""
        pool = await self._get_pool()

        query = """
            SELECT
                ts.*,
                ta.id as target_agent_id, ta.name as target_agent_name,
                ta.websocket_url, ta.sample_rate, ta.encoding,
                ua.id as user_agent_id, ua.name as user_agent_name,
                ua.system_prompt, ua.evaluation_criteria, ua.model_config
            FROM test_suites ts
            LEFT JOIN target_agents ta ON ts.target_agent_id = ta.id
            LEFT JOIN user_agents ua ON ts.user_agent_id = ua.id
            WHERE ts.id = $1
        """

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchrow(query, suite_id)
                if not result:
                    return None

                # Build the response
                suite_data = dict(result)

                # Extract target agent data
                target_agent = None
                if suite_data.get("target_agent_id"):
                    target_agent = TargetAgent(
                        id=suite_data["target_agent_id"],
                        user_id=UUID("00000000-0000-0000-0000-000000000000"),  # Not needed for response
                        name=suite_data["target_agent_name"],
                        websocket_url=suite_data["websocket_url"],
                        sample_rate=suite_data["sample_rate"],
                        encoding=suite_data["encoding"],
                        created_at=suite_data["created_at"],  # Will be overridden by suite
                        updated_at=suite_data["updated_at"]   # Will be overridden by suite
                    )

                # Extract user agent data
                user_agent = None
                if suite_data.get("user_agent_id"):
                    user_agent = UserAgent(
                        id=suite_data["user_agent_id"],
                        user_id=UUID("00000000-0000-0000-0000-000000000000"),  # Not needed for response
                        name=suite_data["user_agent_name"],
                        system_prompt=suite_data["system_prompt"],
                        evaluation_criteria=suite_data["evaluation_criteria"],
                        model_config=suite_data["model_config"],
                        created_at=suite_data["created_at"],  # Will be overridden by suite
                        updated_at=suite_data["updated_at"]   # Will be overridden by suite
                    )

                # Get test cases
                test_cases = await self._get_test_cases_for_suite(suite_id)

                return TestSuiteWithRelations(
                    id=suite_data["id"],
                    user_id=suite_data["user_id"],
                    name=suite_data["name"],
                    description=suite_data["description"],
                    target_agent_id=suite_data["target_agent_id"],
                    user_agent_id=suite_data["user_agent_id"],
                    created_at=suite_data["created_at"],
                    updated_at=suite_data["updated_at"],
                    target_agent=target_agent,
                    user_agent=user_agent,
                    test_cases=test_cases
                )
        except Exception as e:
            logger.error(f"Error getting test suite with relations: {e}")
            raise

    async def _get_test_cases_for_suite(self, suite_id: UUID) -> List[TestCase]:
        """Get test cases for a suite."""
        pool = await self._get_pool()

        query = """
            SELECT * FROM test_cases
            WHERE test_suite_id = $1 AND is_active = true
            ORDER BY order_index, created_at
        """

        try:
            async with pool.acquire() as conn:
                results = await conn.fetch(query, suite_id)
                return [TestCase(**dict(row)) for row in results]
        except Exception as e:
            logger.error(f"Error getting test cases for suite {suite_id}: {e}")
            return []

    async def get_test_suites_by_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[TestSuite]:
        """Get test suites for a user."""
        results = await self.get_all_by_user(user_id, limit, offset)
        return [TestSuite(**result) for result in results]

    async def update_test_suite(self, suite_id: UUID, data: TestSuiteUpdate) -> bool:
        """Update a test suite."""
        update_data = data.model_dump(exclude_unset=True)
        return await self.update(suite_id, update_data)

    async def delete_test_suite(self, suite_id: UUID) -> bool:
        """Delete a test suite."""
        return await self.delete(suite_id)

    async def get_test_suite_count(self, user_id: UUID) -> int:
        """Get count of test suites for a user."""
        return await self.count_by_user(user_id)
