"""
Test case CRUD service.

This module provides CRUD operations for test cases.
"""

from typing import List, Optional
from uuid import UUID

from services.database_service import DatabaseService
from models.test_suite_models import TestCaseCreate, TestCaseUpdate, TestCase
from telemetrics.logger import logger


class TestCaseService(DatabaseService[TestCase]):
    """Service for test case CRUD operations."""

    def __init__(self):
        super().__init__("test_cases")

    async def create_test_case(self, data: TestCaseCreate) -> UUID:
        """Create a new test case."""
        case_data = data.model_dump()
        return await self.create(case_data)

    async def get_test_case(self, case_id: UUID) -> Optional[TestCase]:
        """Get a test case by ID."""
        result = await self.get_by_id(case_id)
        if result:
            return TestCase(**result)
        return None

    async def get_test_cases_by_suite(
        self, suite_id: UUID, include_inactive: bool = False, limit: int = 100, offset: int = 0
    ) -> List[TestCase]:
        """Get test cases for a test suite."""
        pool = await self._get_pool()

        active_filter = "" if include_inactive else "AND is_active = true"

        query = f"""
            SELECT * FROM test_cases
            WHERE test_suite_id = $1 {active_filter}
            ORDER BY order_index, created_at
            LIMIT $2 OFFSET $3
        """

        try:
            async with pool.acquire() as conn:
                results = await conn.fetch(query, suite_id, limit, offset)
                return [TestCase(**dict(row)) for row in results]
        except Exception as e:
            logger.error(f"Error fetching test cases for suite {suite_id}: {e}")
            raise

    async def update_test_case(self, case_id: UUID, data: TestCaseUpdate) -> bool:
        """Update a test case."""
        update_data = data.model_dump(exclude_unset=True)
        return await self.update(case_id, update_data)

    async def delete_test_case(self, case_id: UUID) -> bool:
        """Delete a test case."""
        return await self.delete(case_id)

    async def reorder_test_cases(self, suite_id: UUID, case_orders: List[dict]) -> bool:
        """Reorder test cases within a suite."""
        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for case_order in case_orders:
                        await conn.execute(
                            "UPDATE test_cases SET order_index = $1 WHERE id = $2 AND test_suite_id = $3",
                            case_order["order_index"],
                            case_order["case_id"],
                            suite_id
                        )
            logger.info(f"Reordered test cases for suite {suite_id}")
            return True
        except Exception as e:
            logger.error(f"Error reordering test cases for suite {suite_id}: {e}")
            return False

    async def get_test_case_count(self, suite_id: UUID, include_inactive: bool = False) -> int:
        """Get count of test cases for a suite."""
        pool = await self._get_pool()

        active_filter = "" if include_inactive else "AND is_active = true"

        query = f"SELECT COUNT(*) FROM test_cases WHERE test_suite_id = $1 {active_filter}"

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(query, suite_id)
                return result or 0
        except Exception as e:
            logger.error(f"Error counting test cases for suite {suite_id}: {e}")
            raise
