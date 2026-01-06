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
        supabase_client = await self._get_client()

        try:
            filters = {"test_suite_id": str(suite_id)}
            if not include_inactive:
                filters["is_active"] = True

            results = await supabase_client.select(
                "test_cases",
                filters=filters,
                order_by="order_index,created_at",
                limit=limit,
                offset=offset
            )

            if not results:
                return []

            return [TestCase(**tc_data) for tc_data in results]
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
        supabase_client = await self._get_client()

        try:
            for case_order in case_orders:
                await supabase_client.update(
                    "test_cases",
                    {"id": case_order["case_id"], "test_suite_id": str(suite_id)},
                    {"order_index": case_order["order_index"]}
                )
            logger.info(f"Reordered test cases for suite {suite_id}")
            return True
        except Exception as e:
            logger.error(f"Error reordering test cases for suite {suite_id}: {e}")
            return False

    async def get_test_case_count(self, suite_id: UUID, include_inactive: bool = False) -> int:
        """Get count of test cases for a suite."""
        supabase_client = await self._get_client()

        try:
            filters = {"test_suite_id": str(suite_id)}
            if not include_inactive:
                filters["is_active"] = True

            results = await supabase_client.select(
                "test_cases",
                filters=filters
            )

            return len(results) if results else 0
        except Exception as e:
            logger.error(f"Error counting test cases for suite {suite_id}: {e}")
            raise

    async def delete_test_cases_by_suite_id(self, suite_id: UUID) -> int:
        """Delete all test cases associated with a test suite ID."""
        supabase_client = await self._get_client()

        try:
            # First get all test cases with this test_suite_id
            test_cases = await supabase_client.select(
                self.table_name,
                filters={'test_suite_id': str(suite_id)}
            )

            deleted_count = 0
            if test_cases:
                for case in test_cases:
                    try:
                        # Delete each test case
                        await supabase_client.delete(self.table_name, {'id': case['id']})
                        deleted_count += 1
                        logger.info(f"Deleted test case {case['id']} associated with test suite {suite_id}")
                    except Exception as e:
                        logger.error(f"Failed to delete test case {case['id']}: {e}")

            logger.info(f"Deleted {deleted_count} test cases associated with test suite {suite_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting test cases by test suite ID: {e}")
            return 0
