"""
Test suite CRUD service.

This module provides CRUD operations for test suites and related entities.
"""

import json
from typing import List, Optional, Dict, Any
from uuid import UUID

from services.database_service import DatabaseService
from services.test_case_service import TestCaseService
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
        # Only include basic fields that exist in the database
        suite_data = {
            "user_id": str(user_id),
            "name": data.name,
            "description": data.description
        }
        # Only add optional fields if they exist and are not None
        if data.target_agent_id:
            suite_data["target_agent_id"] = str(data.target_agent_id)
        if data.user_agent_id:
            suite_data["user_agent_id"] = str(data.user_agent_id)
        return await self.create(suite_data)

    async def get_test_suite(self, suite_id: UUID) -> Optional[TestSuite]:
        """Get a test suite by ID."""
        result = await self.get_by_id(suite_id)
        if result:
            return TestSuite(**result)
        return None

    async def get_test_suite_with_relations(self, suite_id: UUID) -> Optional[TestSuiteWithRelations]:
        """Get a test suite with all related entities."""
        supabase_client = await self._get_client()

        try:
            # First get the test suite
            suite_result = await supabase_client.select(
                "test_suites",
                filters={"id": str(suite_id)}
            )

            if not suite_result or len(suite_result) == 0:
                return None

            suite_data = suite_result[0]

            # Get target agent if exists
            target_agent = None
            if suite_data.get('target_agent_id'):
                target_agent_result = await supabase_client.select(
                    "target_agents",
                    filters={"id": suite_data['target_agent_id']}
                )
                if target_agent_result and len(target_agent_result) > 0:
                    ta_data = target_agent_result[0]
                    target_agent = TargetAgent(
                        id=ta_data['id'],
                        user_id=ta_data['user_id'],
                        name=ta_data['name'],
                        websocket_url=ta_data['websocket_url'],
                        sample_rate=ta_data['sample_rate'],
                        encoding=ta_data['encoding'],
                        created_at=ta_data['created_at'],
                        updated_at=ta_data['updated_at']
                    )

            # Get user agent if exists
            user_agent = None
            if suite_data.get('user_agent_id'):
                user_agent_result = await supabase_client.select(
                    "user_agents",
                    filters={"id": suite_data['user_agent_id']}
                )
                if user_agent_result and len(user_agent_result) > 0:
                    ua_data = user_agent_result[0]
                    # Parse JSON fields
                    evaluation_criteria = ua_data.get('evaluation_criteria')
                    if isinstance(evaluation_criteria, str):
                        evaluation_criteria = json.loads(evaluation_criteria)

                    model_config = ua_data.get('agent_model_config')
                    if isinstance(model_config, str):
                        model_config = json.loads(model_config)

                    user_agent = UserAgent(
                        id=ua_data['id'],
                        user_id=ua_data['user_id'],
                        name=ua_data['name'],
                        system_prompt=ua_data['system_prompt'],
                        evaluation_criteria=evaluation_criteria,
                        agent_model_config=model_config,
                        pranthora_agent_id=ua_data.get('pranthora_agent_id'),
                        created_at=ua_data['created_at'],
                        updated_at=ua_data['updated_at']
                    )

            # Get individual test case statuses first
            test_case_statuses = await self._get_test_case_statuses(suite_id)

            # Get test cases with their statuses
            test_cases = await self._get_test_cases_for_suite(suite_id, test_case_statuses)

            # Get suite status from all test runs
            suite_status = await self._get_suite_status(suite_id)

            return TestSuiteWithRelations(
                id=suite_data['id'],
                user_id=suite_data['user_id'],
                name=suite_data['name'],
                description=suite_data['description'],
                target_agent_id=suite_data.get('target_agent_id'),
                user_agent_id=suite_data.get('user_agent_id'),
                created_at=suite_data['created_at'],
                updated_at=suite_data['updated_at'],
                target_agent=target_agent,
                user_agent=user_agent,
                test_cases=test_cases,
                suite_status=suite_status
            )

        except Exception as e:
            logger.error(f"Error getting test suite with relations: {e}")
            raise

    async def _get_test_cases_for_suite(self, suite_id: UUID, test_case_statuses: Dict[str, str] = None) -> List[TestCase]:
        """Get test cases for a suite with their current status."""
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                return []

            async_client = await acreate_client(supabase_url, supabase_key)

            # Query test cases directly
            result = await async_client.table('test_cases').select('*').eq(
                'test_suite_id', str(suite_id)
            ).eq('is_active', True).order('order_index').execute()

            if not result.data:
                return []

            # Add status to each test case
            test_cases = []
            for tc_data in result.data:
                tc_id = str(tc_data['id'])
                status = test_case_statuses.get(tc_id, "pending") if test_case_statuses else "pending"
                tc_data['status'] = status
                test_cases.append(TestCase(**tc_data))

            return test_cases
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

    async def nullify_user_agent_references(self, user_agent_id: UUID) -> int:
        """Set user_agent_id to null for all test suites associated with a user agent ID."""
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            # Get database config directly
            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be configured")

            # Create raw Supabase client directly
            async_client = await acreate_client(supabase_url, supabase_key)

            # Use raw Supabase client to do bulk update
            result = await async_client.table(self.table_name).update({
                'user_agent_id': None,
                'updated_at': 'now()'
            }).eq('user_agent_id', str(user_agent_id)).execute()

            updated_count = len(result.data) if result.data else 0
            logger.info(f"Set user_agent_id to null for {updated_count} test suites associated with user agent {user_agent_id}")
            return updated_count

        except Exception as e:
            logger.error(f"Error nullifying user agent references in test suites: {e}")
            return 0

    async def nullify_target_agent_references(self, target_agent_id: UUID) -> int:
        """Set target_agent_id to null for all test suites associated with a target agent ID."""
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            # Get database config directly
            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be configured")

            # Create raw Supabase client directly
            async_client = await acreate_client(supabase_url, supabase_key)

            # Use raw Supabase client to do bulk update
            result = await async_client.table(self.table_name).update({
                'target_agent_id': None,
                'updated_at': 'now()'
            }).eq('target_agent_id', str(target_agent_id)).execute()

            updated_count = len(result.data) if result.data else 0
            logger.info(f"Set target_agent_id to null for {updated_count} test suites associated with target agent {target_agent_id}")
            return updated_count

        except Exception as e:
            logger.error(f"Error nullifying target agent references in test suites: {e}")
            return 0

    async def nullify_test_run_references(self, test_suite_id: UUID) -> int:
        """Set test_suite_id to null for all test runs associated with a test suite ID."""
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            # Get database config directly
            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                raise ValueError("Supabase URL and Key must be configured")

            # Create raw Supabase client directly
            async_client = await acreate_client(supabase_url, supabase_key)

            # Update test_run_history to set test_suite_id to null where it matches
            result = await async_client.table('test_run_history').update(
                {'test_suite_id': None}
            ).eq('test_suite_id', str(test_suite_id)).execute()

            updated_count = len(result.data) if result.data else 0
            if updated_count > 0:
                logger.info(f"Nullified test_suite_id references for {updated_count} test runs")

            return updated_count

        except Exception as e:
            logger.error(f"Error nullifying test run references in test_run_history: {e}")
            return 0

    async def delete_test_suite(self, suite_id: UUID) -> bool:
        """Delete a test suite (deletes all associated test cases and nullifies test runs first)."""
        # Delete all test cases associated with this test suite first
        test_case_service = TestCaseService()
        try:
            deleted_cases = await test_case_service.delete_test_cases_by_suite_id(suite_id)
            if deleted_cases > 0:
                logger.info(f"Deleted {deleted_cases} test cases associated with test suite {suite_id}")
        except Exception as e:
            logger.error(f"Error deleting test cases for test suite {suite_id}: {e}")
        finally:
            await test_case_service.close()

        # Nullify all test runs associated with this test suite
        nullified_runs = await self.nullify_test_run_references(suite_id)
        if nullified_runs > 0:
            logger.info(f"Nullified test_suite_id references for {nullified_runs} test runs")

        # Now delete the test suite
        return await self.delete(suite_id)

    async def get_test_suite_count(self, user_id: UUID) -> int:
        """Get count of test suites for a user."""
        return await self.count_by_user(user_id)

    async def _get_suite_status(self, suite_id: UUID) -> Optional[str]:
        """
        Get suite status based on latest test run for the suite.
        
        Status values: pending, running, failed, completed
        """
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                return None

            async_client = await acreate_client(supabase_url, supabase_key)

            # Get latest test run for this suite
            result = await async_client.table('test_run_history').select(
                'status'
            ).eq('test_suite_id', str(suite_id)).order('created_at', desc=True).limit(1).execute()

            if not result.data:
                return None  # No test runs = no status

            # Return status directly from latest run
            return result.data[0].get('status', 'pending')

        except Exception as e:
            logger.error(f"Error getting suite status for {suite_id}: {e}")
            return None

    async def _get_test_case_statuses(self, suite_id: UUID) -> Dict[str, str]:
        """
        Get latest status for each test case from test_case_results.
        Status values: pending, running, failed, completed
        """
        try:
            from supabase.client import acreate_client
            from static_memory_cache import StaticMemoryCache

            db_config = StaticMemoryCache.get_database_config()
            supabase_url = db_config.get("supabase_url")
            supabase_key = db_config.get("supabase_key")

            if not supabase_url or not supabase_key:
                return {}

            async_client = await acreate_client(supabase_url, supabase_key)

            # Get all results for this suite ordered by created_at desc
            results = await async_client.table('test_case_results').select(
                'test_case_id, status'
            ).eq('test_suite_id', str(suite_id)).order('created_at', desc=True).execute()

            if not results.data:
                return {}

            # Get latest status for each test case (first occurrence due to desc order)
            statuses = {}
            for record in results.data:
                tc_id = record.get('test_case_id')
                if tc_id and tc_id not in statuses:
                    statuses[tc_id] = record.get('status', 'pending')

            return statuses

        except Exception as e:
            logger.error(f"Error getting test case statuses: {e}")
            return {}
