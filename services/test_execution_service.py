"""
Test execution service for running test cases and test suites.

This module provides functionality to execute individual test cases or entire test suites,
simulate conversations, evaluate results, and store execution history.
"""

import asyncio
import json
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from services.database_service import DatabaseService
from services.test_case_service import TestCaseService
from services.test_suite_service import TestSuiteService
from services.target_agent_service import TargetAgentService
from services.user_agent_service import UserAgentService
from services.test_history_service import TestRunHistoryService, TestCaseResultService
from services.pranthora_api_client import PranthoraApiClient
from models.test_suite_models import (
    TestSuite, TestCase, TestRunHistory, TestCaseResult,
    TestRunHistoryBase, TestCaseResultBase
)
from telemetrics.logger import logger


class TestExecutionService:
    """Service for executing test cases and test suites."""

    def __init__(self):
        self.database_service = DatabaseService("test_run_history")
        self.test_case_service = TestCaseService()
        self.test_suite_service = TestSuiteService()
        self.target_agent_service = TargetAgentService()
        self.user_agent_service = UserAgentService()
        self.test_run_service = TestRunHistoryService()
        self.test_result_service = TestCaseResultService()
        self.pranthora_client = PranthoraApiClient()

    async def run_test_suite(
        self,
        test_suite_id: UUID,
        user_id: UUID,
        concurrent_calls: Optional[int] = None
    ) -> UUID:
        """
        Run all active test cases in a test suite.

        Args:
            test_suite_id: ID of the test suite to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)

        Returns:
            UUID of the created test run
        """
        try:
            # Validate test suite exists and user has access
            test_suite = await self.test_suite_service.get_test_suite(test_suite_id)
            if not test_suite:
                raise ValueError(f"Test suite {test_suite_id} not found")

            if test_suite.user_id != user_id:
                raise ValueError(f"User {user_id} does not have access to test suite {test_suite_id}")

            # Get active test cases for the suite
            test_cases = await self.test_case_service.get_test_cases_by_suite(
                test_suite_id, include_inactive=False
            )

            if not test_cases:
                raise ValueError(f"No active test cases found in test suite {test_suite_id}")

            # Create test run record
            test_run_id = await self._create_test_run(test_suite_id, user_id, len(test_cases))

            # Execute test cases asynchronously
            asyncio.create_task(
                self._execute_test_cases_async(test_run_id, test_cases, concurrent_calls)
            )

            logger.info(f"Started test suite execution: {test_run_id} with {len(test_cases)} test cases")
            return test_run_id

        except Exception as e:
            logger.error(f"Error starting test suite execution: {e}")
            raise

    async def run_single_test_case(
        self,
        test_case_id: UUID,
        user_id: UUID,
        concurrent_calls: Optional[int] = None
    ) -> UUID:
        """
        Run a single test case.

        Args:
            test_case_id: ID of the test case to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)

        Returns:
            UUID of the created test run
        """
        try:
            # Get test case and validate access
            test_case = await self.test_case_service.get_test_case(test_case_id)
            if not test_case:
                raise ValueError(f"Test case {test_case_id} not found")

            # Get test suite to validate user access
            test_suite = await self.test_suite_service.get_test_suite(test_case.test_suite_id)
            if not test_suite or test_suite.user_id != user_id:
                raise ValueError(f"User {user_id} does not have access to test case {test_case_id}")

            # Create test run record
            test_run_id = await self._create_test_run(test_case.test_suite_id, user_id, 1)

            # Execute single test case asynchronously
            asyncio.create_task(
                self._execute_single_test_case_async(test_run_id, test_case, concurrent_calls)
            )

            logger.info(f"Started single test case execution: {test_run_id}")
            return test_run_id

        except Exception as e:
            logger.error(f"Error starting single test case execution: {e}")
            raise

    async def _create_test_run(self, test_suite_id: UUID, user_id: UUID, total_cases: int) -> UUID:
        """Create a new test run record."""
        run_data = {
            "test_suite_id": str(test_suite_id),
            "user_id": str(user_id),
            "status": "running",
            "total_test_cases": total_cases,
            "passed_count": 0,
            "failed_count": 0,
            "alert_count": 0,
            "started_at": datetime.utcnow().isoformat(),
        }

        run_id = await self.database_service.create(run_data)
        logger.info(f"Created test run: {run_id}")
        return run_id

    async def _execute_test_cases_async(
        self,
        test_run_id: UUID,
        test_cases: List[TestCase],
        concurrent_calls: Optional[int] = None
    ):
        """Execute multiple test cases asynchronously."""
        try:
            results = []
            passed_count = 0
            failed_count = 0
            alert_count = 0

            # Execute test cases sequentially for now (can be made concurrent later)
            for test_case in test_cases:
                try:
                    result = await self._execute_test_case(test_case, test_run_id, concurrent_calls)
                    results.append(result)

                    if result["status"] == "pass":
                        passed_count += 1
                    elif result["status"] == "fail":
                        failed_count += 1
                    elif result["status"] == "alert":
                        alert_count += 1

                except Exception as e:
                    logger.error(f"Error executing test case {test_case.id}: {e}")
                    failed_count += 1

            # Update test run with final results
            await self._update_test_run_status(
                test_run_id, "completed", passed_count, failed_count, alert_count
            )

            logger.info(f"Completed test run {test_run_id}: {passed_count} passed, {failed_count} failed, {alert_count} alerts")

        except Exception as e:
            logger.error(f"Error in test case execution: {e}")
            await self._update_test_run_status(test_run_id, "failed", 0, 0, 0)

    async def _execute_single_test_case_async(
        self,
        test_run_id: UUID,
        test_case: TestCase,
        concurrent_calls: Optional[int] = None
    ):
        """Execute a single test case asynchronously."""
        try:
            result = await self._execute_test_case(test_case, test_run_id, concurrent_calls)

            passed_count = 1 if result["status"] == "pass" else 0
            failed_count = 1 if result["status"] == "fail" else 0
            alert_count = 1 if result["status"] == "alert" else 0

            await self._update_test_run_status(
                test_run_id, "completed", passed_count, failed_count, alert_count
            )

            logger.info(f"Completed single test case execution {test_run_id}: {result['status']}")

        except Exception as e:
            logger.error(f"Error in single test case execution: {e}")
            await self._update_test_run_status(test_run_id, "failed", 0, 1, 0)

    async def _execute_test_case(
        self,
        test_case: TestCase,
        test_run_id: UUID,
        concurrent_calls: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a single test case."""
        try:
            logger.info(f"Executing test case: {test_case.id} - {test_case.name}")

            # Get test suite and related agents
            test_suite = await self.test_suite_service.get_test_suite(test_case.test_suite_id)
            if not test_suite:
                raise ValueError(f"Test suite {test_case.test_suite_id} not found")

            # Validate agents exist
            target_agent = None
            if test_suite.target_agent_id:
                target_agent = await self.target_agent_service.get_target_agent(test_suite.target_agent_id)

            user_agent = None
            if test_suite.user_agent_id:
                user_agent = await self.user_agent_service.get_user_agent(test_suite.user_agent_id)

            if not target_agent:
                raise ValueError(f"Target agent not found for test suite {test_case.test_suite_id}")

            if not user_agent:
                raise ValueError(f"User agent not found for test suite {test_case.test_suite_id}")

            # Simulate conversation using goals/prompts
            conversation_result = await self._simulate_conversation(
                test_case, target_agent, user_agent, concurrent_calls or test_case.default_concurrent_calls
            )

            # Evaluate results based on evaluation criteria
            evaluation_result = await self._evaluate_test_results(
                test_case, conversation_result, user_agent
            )

            # Determine final status
            status = self._determine_test_status(evaluation_result)

            # Store test case result
            result_id = await self.test_result_service.create_test_case_result_with_recording(
                test_run_id=test_run_id,
                test_case_id=test_case.id,
                test_suite_id=test_case.test_suite_id,
                status=status,
                conversation_logs=conversation_result.get("conversation_logs", []),
                evaluation_result=evaluation_result,
                error_message=conversation_result.get("error_message"),
                pcm_frames=conversation_result.get("audio_data", b"")
            )

            return {
                "result_id": result_id,
                "status": status,
                "conversation_result": conversation_result,
                "evaluation_result": evaluation_result
            }

        except Exception as e:
            logger.error(f"Error executing test case {test_case.id}: {e}")
            # Store failed result
            try:
                result_id = await self.test_result_service.create_test_case_result_with_recording(
                    test_run_id=test_run_id,
                    test_case_id=test_case.id,
                    test_suite_id=test_case.test_suite_id,
                    status="fail",
                    error_message=str(e),
                    pcm_frames=b""
                )
                return {"result_id": result_id, "status": "fail", "error": str(e)}
            except Exception as store_error:
                logger.error(f"Error storing failed test result: {store_error}")
                return {"status": "fail", "error": str(e)}

    async def _simulate_conversation(
        self,
        test_case: TestCase,
        target_agent,
        user_agent,
        concurrent_calls: int
    ) -> Dict[str, Any]:
        """Simulate a conversation using the test case goals."""
        try:
            conversation_logs = []
            audio_data = b""

            # For now, simulate a basic conversation
            # In a real implementation, this would connect to the actual agents
            # and execute the goals/prompts

            logger.info(f"Simulating conversation for test case {test_case.id}")

            # Mock conversation simulation
            for i, goal in enumerate(test_case.goals):
                conversation_logs.append({
                    "turn": i + 1,
                    "type": "user_input",
                    "content": goal.get("text", ""),
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Simulate agent response
                conversation_logs.append({
                    "turn": i + 1,
                    "type": "agent_response",
                    "content": f"Simulated response to: {goal.get('text', '')}",
                    "timestamp": datetime.utcnow().isoformat()
                })

            # Generate mock audio data (empty for now)
            # In real implementation, this would be actual conversation audio

            return {
                "conversation_logs": conversation_logs,
                "audio_data": audio_data,
                "duration_seconds": len(test_case.goals) * 5,  # Mock duration
                "success": True
            }

        except Exception as e:
            logger.error(f"Error simulating conversation: {e}")
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "error_message": str(e),
                "success": False
            }

    async def _evaluate_test_results(
        self,
        test_case: TestCase,
        conversation_result: Dict[str, Any],
        user_agent
    ) -> Dict[str, Any]:
        """Evaluate test results based on evaluation criteria."""
        try:
            evaluation_result = {
                "criteria_evaluated": [],
                "overall_score": 0.0,
                "passed_criteria": 0,
                "total_criteria": len(test_case.evaluation_criteria),
                "evaluation_details": []
            }

            # For now, provide mock evaluation
            # In real implementation, this would use the user agent to evaluate
            # the conversation against the evaluation criteria

            for i, criterion in enumerate(test_case.evaluation_criteria):
                criterion_result = {
                    "criterion_id": i + 1,
                    "type": criterion.get("type", "unknown"),
                    "expected": criterion.get("expected", ""),
                    "passed": True,  # Mock pass for now
                    "score": 1.0,
                    "details": f"Mock evaluation of criterion {i + 1}"
                }

                evaluation_result["criteria_evaluated"].append(criterion_result)
                if criterion_result["passed"]:
                    evaluation_result["passed_criteria"] += 1
                    evaluation_result["overall_score"] += criterion_result["score"]

            if evaluation_result["total_criteria"] > 0:
                evaluation_result["overall_score"] /= evaluation_result["total_criteria"]

            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating test results: {e}")
            return {
                "error": str(e),
                "overall_score": 0.0,
                "passed_criteria": 0,
                "total_criteria": len(test_case.evaluation_criteria)
            }

    def _determine_test_status(self, evaluation_result: Dict[str, Any]) -> str:
        """Determine the overall test status based on evaluation results."""
        try:
            total_criteria = evaluation_result.get("total_criteria", 0)
            passed_criteria = evaluation_result.get("passed_criteria", 0)
            overall_score = evaluation_result.get("overall_score", 0.0)

            if total_criteria == 0:
                return "pass"  # No criteria to evaluate

            # If all criteria pass and score is good, mark as pass
            if passed_criteria == total_criteria and overall_score >= 0.7:
                return "pass"
            elif passed_criteria > 0:  # Some criteria passed
                return "alert"
            else:  # No criteria passed
                return "fail"

        except Exception as e:
            logger.error(f"Error determining test status: {e}")
            return "fail"

    async def _update_test_run_status(
        self,
        test_run_id: UUID,
        status: str,
        passed_count: int,
        failed_count: int,
        alert_count: int
    ):
        """Update the test run status and statistics."""
        try:
            update_data = {
                "status": status,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "alert_count": alert_count
            }

            if status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow().isoformat()

            success = await self.database_service.update(test_run_id, update_data)
            if success:
                logger.info(f"Updated test run {test_run_id} status to {status}")
            else:
                logger.error(f"Failed to update test run {test_run_id}")

        except Exception as e:
            logger.error(f"Error updating test run status: {e}")

    async def close(self):
        """Close all service connections."""
        await self.database_service.close()
        await self.test_case_service.close()
        await self.test_suite_service.close()
