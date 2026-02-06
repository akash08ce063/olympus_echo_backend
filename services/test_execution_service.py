"""
Test execution service for running test cases and test suites.

This module provides functionality to execute individual test cases or entire test suites,
simulate conversations, evaluate results, and store execution history.
"""

import asyncio
import time
import uuid
import wave
import audioop
import io
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

from services.database_service import DatabaseService
from services.test_case_service import TestCaseService
from services.test_suite_service import TestSuiteService
from services.agent_connection_manager import AgentConnectionManager
from services.target_connection_factory import create_target_connection
from services.target_agent_service import TargetAgentService
from services.user_agent_service import UserAgentService
from services.test_history_service import TestRunHistoryService, TestCaseResultService
from services.recording_storage_service import RecordingStorageService
from services.pranthora_api_client import PranthoraApiClient
from services.evaluation_agent_service import EvaluationAgentService
from models.test_suite_models import TestCase
from telemetrics.logger import logger


class TranscriptMessage(BaseModel):
    """Pydantic model for transcript message."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class EvaluationRequest(BaseModel):
    """Pydantic model for evaluation request."""

    test_case_id: UUID
    test_run_id: UUID
    request_ids: List[str] = Field(..., description="Request IDs to fetch transcripts")
    goals: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_criteria: List[Dict[str, Any]] = Field(default_factory=list)
    test_case_name: str = "Test Case"


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
        self.recording_service = RecordingStorageService()
        self.pranthora_client = PranthoraApiClient()
        self.evaluation_service = EvaluationAgentService()

        # Recording setup
        self.sample_rate = 8000  # μ-law sample rate

        # Read chunk duration from config
        from static_memory_cache import StaticMemoryCache

        chunk_duration_ms = StaticMemoryCache.get_audio_chunk_duration_ms()
        self.chunk_duration_seconds = chunk_duration_ms / 1000.0
        self.connection_sync_timeout = StaticMemoryCache.get_connection_sync_timeout_seconds()

    async def run_test_suite(
        self,
        test_suite_id: UUID,
        user_id: UUID,
        concurrent_calls: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
        execution_mode: str = "sequential",
    ) -> UUID:
        """
        Run all active test cases in a test suite.

        Args:
            test_suite_id: ID of the test suite to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)
            request_ids: List of request IDs from x-pranthora-callid header (comma-separated, one per concurrent call)

        Returns:
            UUID of the created test run
        """
        try:
            # Validate test suite exists and user has access
            test_suite = await self.test_suite_service.get_test_suite(test_suite_id)
            if not test_suite:
                raise ValueError(f"Test suite {test_suite_id} not found")

            if test_suite.user_id != user_id:
                raise ValueError(
                    f"User {user_id} does not have access to test suite {test_suite_id}"
                )

            # Get active test cases for the suite
            test_cases = await self.test_case_service.get_test_cases_by_suite(
                test_suite_id, include_inactive=False
            )

            if not test_cases:
                raise ValueError(f"No active test cases found in test suite {test_suite_id}")

            # Use first request_id for test run creation (backward compatibility)
            primary_request_id = request_ids[0] if request_ids and len(request_ids) > 0 else None
            # Create test run record
            test_run_id = await self._create_test_run(
                test_suite_id, user_id, len(test_cases), primary_request_id
            )

            # Execute test cases asynchronously
            asyncio.create_task(
                self._execute_test_cases_async(
                    test_run_id, test_cases, concurrent_calls, execution_mode, request_ids
                )
            )

            logger.info(
                f"Started test suite execution: {test_run_id} with {len(test_cases)} test cases"
            )
            return test_run_id

        except Exception as e:
            logger.error(f"Error starting test suite execution: {e}")
            raise

    async def run_single_test_case(
        self,
        test_case_id: UUID,
        user_id: UUID,
        concurrent_calls: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
    ) -> UUID:
        """
        Run a single test case.

        Args:
            test_case_id: ID of the test case to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)
            request_ids: List of request IDs from x-pranthora-callid header (comma-separated, one per concurrent call)

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

            # Use first request_id for test run creation (backward compatibility)
            primary_request_id = request_ids[0] if request_ids and len(request_ids) > 0 else None
            # Create test run record
            test_run_id = await self._create_test_run(
                test_case.test_suite_id, user_id, 1, primary_request_id
            )

            # Execute single test case asynchronously
            asyncio.create_task(
                self._execute_single_test_case_async(
                    test_run_id, test_case, concurrent_calls, request_ids
                )
            )

            logger.info(f"Started single test case execution: {test_run_id}")
            return test_run_id

        except Exception as e:
            logger.error(f"Error starting single test case execution: {e}")
            raise

    async def _create_test_run(
        self, test_suite_id: UUID, user_id: UUID, total_cases: int, request_id: Optional[str] = None
    ) -> UUID:
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

        # If request_id is provided and it's a valid UUID, use it as the primary key
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
        )

        if request_id and uuid_pattern.match(request_id):
            run_data["id"] = request_id
            # Use create_with_id instead of create to specify the ID
            run_id = await self.database_service.create_with_id(run_data)
        else:
            run_id = await self.database_service.create(run_data)

        logger.info(f"Created test run: {run_id}")
        return run_id

    async def _execute_test_cases_async(
        self,
        test_run_id: UUID,
        test_cases: List[TestCase],
        concurrent_calls: Optional[int] = None,
        execution_mode: str = "sequential",
        request_ids: Optional[List[str]] = None,
    ):
        """Execute multiple test cases asynchronously."""
        try:
            results = []
            passed_count = 0
            failed_count = 0
            alert_count = 0

            if execution_mode == "parallel":
                # Execute all test cases in parallel
                logger.info(f"Executing {len(test_cases)} test cases in parallel mode")
                # Each test case needs its own unique request_ids to avoid primary key conflicts
                # For parallel execution, ignore suite-level request_ids and generate unique ones per test case
                tasks = []
                for test_case in test_cases:
                    # Determine concurrent calls for this specific test case
                    test_case_concurrent_calls = (
                        test_case.default_concurrent_calls
                        if test_case.default_concurrent_calls
                        and test_case.default_concurrent_calls > 0
                        else (concurrent_calls if concurrent_calls and concurrent_calls > 0 else 1)
                    )
                    # Generate unique request_ids for this test case (don't use suite-level request_ids to avoid conflicts)
                    test_case_request_ids = [
                        str(uuid4()) for _ in range(test_case_concurrent_calls)
                    ]
                    logger.info(
                        f"Generated {len(test_case_request_ids)} unique request_ids for test case {test_case.id} (concurrent_calls={test_case_concurrent_calls})"
                    )
                    tasks.append(
                        self._execute_test_case(
                            test_case, test_run_id, concurrent_calls, test_case_request_ids
                        )
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error executing test case {test_cases[i].id}: {result}")
                        failed_count += 1
                    else:
                        # Status values: completed, failed
                        if result.get("status") == "completed":
                            passed_count += 1
                        elif result.get("status") == "failed":
                            failed_count += 1
            else:
                # Execute test cases sequentially (default)
                logger.info(f"Executing {len(test_cases)} test cases in sequential mode")
                for test_case in test_cases:
                    try:
                        # For sequential execution, generate unique request_ids per test case
                        # Determine concurrent calls for this specific test case
                        test_case_concurrent_calls = (
                            test_case.default_concurrent_calls
                            if test_case.default_concurrent_calls
                            and test_case.default_concurrent_calls > 0
                            else (
                                concurrent_calls if concurrent_calls and concurrent_calls > 0 else 1
                            )
                        )
                        # Generate unique request_ids for this test case (don't reuse suite-level request_ids)
                        test_case_request_ids = [
                            str(uuid4()) for _ in range(test_case_concurrent_calls)
                        ]
                        logger.info(
                            f"Generated {len(test_case_request_ids)} unique request_ids for test case {test_case.id} (concurrent_calls={test_case_concurrent_calls})"
                        )
                        result = await self._execute_test_case(
                            test_case, test_run_id, concurrent_calls, test_case_request_ids
                        )
                        results.append(result)

                        # Status values: completed, failed
                        if result["status"] == "completed":
                            passed_count += 1
                        elif result["status"] == "failed":
                            failed_count += 1

                    except Exception as e:
                        logger.error(f"Error executing test case {test_case.id}: {e}")
                        failed_count += 1

            # Update test run with final results
            # Set status to "failed" if any test failed, otherwise "completed"
            final_status = "failed" if failed_count > 0 else "completed"
            await self._update_test_run_status(
                test_run_id, final_status, passed_count, failed_count, alert_count
            )

            logger.info(
                f"Completed test run {test_run_id}: status={final_status}, {passed_count} passed, {failed_count} failed, {alert_count} alerts (mode: {execution_mode})"
            )

        except Exception as e:
            logger.error(f"Error in test case execution: {e}")
            await self._update_test_run_status(test_run_id, "failed", 0, 0, 0)

    async def _execute_single_test_case_async(
        self,
        test_run_id: UUID,
        test_case: TestCase,
        concurrent_calls: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
    ):
        """Execute a single test case asynchronously."""
        try:
            result = await self._execute_test_case(
                test_case, test_run_id, concurrent_calls, request_ids
            )

            # Status values: completed, failed
            passed_count = 1 if result["status"] == "completed" else 0
            failed_count = 1 if result["status"] == "failed" else 0
            alert_count = 0

            # For single test case execution, update test run status based on result
            test_run_status = "failed" if result["status"] == "failed" else "completed"
            await self._update_test_run_status(
                test_run_id, test_run_status, passed_count, failed_count, alert_count
            )

            logger.info(
                f"Completed single test case execution {test_run_id}: {result['status']}, run status: {test_run_status}"
            )

        except Exception as e:
            logger.error(f"Error in single test case execution: {e}")
            # Don't update test run status on error either - let external processes handle completion

    async def _execute_test_case(
        self,
        test_case: TestCase,
        test_run_id: UUID,
        concurrent_calls: Optional[int] = None,
        request_ids: Optional[List[str]] = None,
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
                target_agent = await self.target_agent_service.get_target_agent(
                    test_suite.target_agent_id
                )

            user_agent = None
            if test_suite.user_agent_id:
                user_agent = await self.user_agent_service.get_user_agent(test_suite.user_agent_id)

            if not target_agent:
                logger.warning(f"Target agent not found for test suite {test_case.test_suite_id}")
                # For development/testing, allow execution without target agent but fail gracefully
                return {
                    "result_id": None,
                    "status": "failed",
                    "error": f"Target agent not found for test suite {test_case.test_suite_id}",
                }

            if not user_agent:
                logger.warning(f"User agent not found for test suite {test_case.test_suite_id}")
                # For development/testing, allow execution without user agent but fail gracefully
                return {
                    "result_id": None,
                    "status": "failed",
                    "error": f"User agent not found for test suite {test_case.test_suite_id}",
                }

            # Determine number of concurrent calls to use
            if test_case.default_concurrent_calls and test_case.default_concurrent_calls > 0:
                calls_to_use = test_case.default_concurrent_calls
            else:
                calls_to_use = concurrent_calls if concurrent_calls and concurrent_calls > 0 else 1

            # Prepare request_ids: use provided ones or generate if not enough provided
            if request_ids and len(request_ids) >= calls_to_use:
                # Use provided request_ids (frontend sent the correct number)
                call_request_ids = request_ids[:calls_to_use]
                logger.info(
                    f"Using {len(call_request_ids)} request_ids from frontend: {call_request_ids}"
                )
            else:
                # Not enough request_ids provided - generate missing ones
                if request_ids:
                    logger.warning(
                        f"Only {len(request_ids)} request_ids provided but {calls_to_use} concurrent calls needed. Generating missing ones."
                    )
                    call_request_ids = request_ids[:]
                    # Generate remaining request_ids
                    for i in range(len(request_ids), calls_to_use):
                        call_request_ids.append(str(uuid4()))
                else:
                    # No request_ids provided - generate all
                    logger.warning(
                        f"No request_ids provided. Generating {calls_to_use} request_ids."
                    )
                    call_request_ids = [str(uuid4()) for _ in range(calls_to_use)]

            # Create separate test_case_result entries for each concurrent call
            created_result_ids = []
            for call_num, call_request_id in enumerate(call_request_ids, start=1):
                initial_result_data = {
                    "id": call_request_id,  # Use request_id as the test_case_result.id (primary key)
                    "test_run_id": str(test_run_id),
                    "test_case_id": str(test_case.id),
                    "test_suite_id": str(test_case.test_suite_id),
                    "status": "running",
                    "conversation_logs": [],
                    "evaluation_result": None,
                    "error_message": None,
                    "concurrent_calls": calls_to_use,
                }
                try:
                    result_id = await self.test_result_service.create_with_id(initial_result_data)
                    created_result_ids.append(result_id)
                    logger.info(
                        f"Created test case result {result_id} (call {call_num}/{calls_to_use}) with request_id {call_request_id} for test case {test_case.id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create test case result for call {call_num}: {e}")
                    # Continue with other calls even if one fails

            if not created_result_ids:
                raise ValueError("Failed to create any test case result entries")

            # Use first request_id as primary for backward compatibility (for websocket path)
            primary_request_id = call_request_ids[0]

            # Decide execution path based on target agent type
            agent_type = (getattr(target_agent, "agent_type", None) or "custom").lower()

            if agent_type == "phone":
                # Phone tests: delegate to Pranthora call APIs instead of WebSocket bridge
                conversation_result = await self._simulate_phone_conversation(
                    test_case,
                    target_agent,
                    user_agent,
                    calls_to_use,
                )
            else:
                # Websocket / VAPI path (existing behavior)
                conversation_result = await self._simulate_conversation(
                    test_case,
                    target_agent,
                    user_agent,
                    calls_to_use,
                    primary_request_id,
                    call_request_ids,
                )

            # Determine status based on conversation result
            if conversation_result.get("success", False):
                # WebSocket connections successful - conversation completed
                if (
                    conversation_result.get("error_message")
                    and "timeout" in conversation_result.get("error_message", "").lower()
                ):
                    # Conversation timeout - completed
                    status = "completed"
                    logger.info(f"Conversation completed with timeout for test case {test_case.id}")
                else:
                    # Conversation completed successfully
                    status = "completed"
                    logger.info(f"Conversation completed successfully for test case {test_case.id}")
            else:
                # Other errors - failed
                status = "failed"
                logger.error(
                    f"Conversation failed for test case {test_case.id}: {conversation_result.get('error_message', 'Unknown error')}"
                )

            # Get wav_file_ids and request_ids from conversation result (wav_file_ids unused; recordings handled by Pranthora)
            wav_file_ids = conversation_result.get("wav_file_ids", [])
            concurrent_calls_count = conversation_result.get("concurrent_calls", 1)

            # Update all test_case_result entries we created (one per concurrent call)
            # Match each result entry with its corresponding call data
            updated_result_ids = []
            for idx, result_id in enumerate(created_result_ids):
                call_num = idx + 1
                call_request_id = call_request_ids[idx] if idx < len(call_request_ids) else None

                # Get the corresponding wav_file_id and conversation logs for this call
                call_wav_file_id = wav_file_ids[idx] if idx < len(wav_file_ids) else None
                call_recording_url = None

                # Update this specific test_case_result entry
                # Only store recording_metadata in conversation_logs, not verbose audio logs
                conversation_logs = []
                if wav_file_ids:
                    conversation_logs.append(
                        {
                            "type": "recording_metadata",
                            "wav_file_ids": wav_file_ids,
                            "concurrent_calls": concurrent_calls_count,
                        }
                    )

                update_data = {
                    "status": status,
                    "conversation_logs": conversation_logs,  # Only metadata, no verbose logs
                    "evaluation_result": None,  # Will be updated by sequential evaluation
                    "error_message": conversation_result.get("error_message"),
                    "concurrent_calls": concurrent_calls_count,
                }

                # No local recording URL; Pranthora provides recording_url via call_session metadata

                try:
                    success = await self.test_result_service.update(result_id, update_data)
                    if success:
                        updated_result_ids.append(result_id)
                        logger.info(
                            f"Updated test case result {result_id} (call {call_num}) to status {status}"
                        )
                    else:
                        logger.error(
                            f"Failed to update test case result {result_id} (call {call_num})"
                        )
                except Exception as e:
                    logger.error(
                        f"Error updating test case result {result_id} (call {call_num}): {e}"
                    )

            # Use first result_id for backward compatibility
            primary_result_id = (
                updated_result_ids[0] if updated_result_ids else created_result_ids[0]
            )

            # Run evaluation sequentially with retries (transcripts need time to persist)
            # Evaluation runs on the first result, but we'll update all entries with the same result
            evaluation_result = await self._evaluate_test_results_sequential(
                test_case, conversation_result, user_agent, test_run_id, primary_result_id
            )
            logger.info(f"Completed evaluation for test case {test_case.id}")

            # Update ALL test_case_result entries with the same evaluation result and status
            # (The primary_result_id was already updated by _evaluate_test_results_sequential)
            if evaluation_result:
                # Determine final status based on evaluation
                final_status = self._determine_test_status(evaluation_result)

                # Update all entries (including primary, which is idempotent) with the same evaluation result and status
                for result_id in updated_result_ids:
                    try:
                        await self.test_result_service.update(
                            result_id,
                            {"status": final_status, "evaluation_result": evaluation_result},
                        )
                        logger.info(
                            f"Updated test case result {result_id} with evaluation result and status {final_status}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error updating test case result {result_id} with evaluation: {e}"
                        )

            return {
                "result_id": primary_result_id,
                "status": status,
                "conversation_result": conversation_result,
                "all_result_ids": updated_result_ids,
            }

        except Exception as e:
            logger.error(f"Error executing test case {test_case.id}: {e}")
            # Update or create failed result
            try:
                # Check if test case result already exists
                existing_result = await self.test_result_service.get_result_by_test_run_and_case(
                    test_run_id, test_case.id
                )

                if existing_result:
                    # Update existing result to failed
                    await self.test_result_service.update(
                        existing_result.id, {"status": "failed", "error_message": str(e)}
                    )
                    result_id = existing_result.id
                else:
                    # Create new failed result
                    result_data = {
                        "test_run_id": str(test_run_id),
                        "test_case_id": str(test_case.id),
                        "test_suite_id": str(test_case.test_suite_id),
                        "status": "failed",
                        "error_message": str(e),
                    }
                    result_id = await self.test_result_service.create(result_data)

                return {"result_id": result_id, "status": "failed", "error": str(e)}
            except Exception as store_error:
                logger.error(f"Error storing failed test result: {store_error}")
                return {"status": "failed", "error": str(e)}

    async def _simulate_conversation(
        self,
        test_case: TestCase,
        target_agent,
        user_agent,
        concurrent_calls: int,
        primary_request_id: Optional[str] = None,
        request_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Simulate conversations by connecting to user agent and target agent websockets,
        bridging audio between them, and recording the conversation(s).

        Supports concurrent calls - creates multiple simultaneous conversations when concurrent_calls > 1.

        Args:
            test_case: Test case to execute
            target_agent: Target agent configuration
            user_agent: User agent configuration
            concurrent_calls: Number of concurrent calls
            primary_request_id: Primary request ID (for backward compatibility)
            request_ids: List of request IDs (one per concurrent call) - preferred over primary_request_id
        """
        try:
            start_time = time.time()

            logger.info(
                f"Starting conversation simulation for test case {test_case.id} with {concurrent_calls} concurrent call(s)"
            )
            logger.info(
                f"Target agent type: {getattr(target_agent, 'agent_type', 'custom')}, User agent: {user_agent.pranthora_agent_id}"
            )

            # Validate agents have required information
            agent_type = (getattr(target_agent, "agent_type", None) or "custom").lower()
            if agent_type == "custom" and not (getattr(target_agent, "websocket_url", None) or "").strip():
                raise ValueError("Target agent (custom) missing websocket_url or HTTP endpoint URL")
            if agent_type == "vapi":
                pc = getattr(target_agent, "provider_config", None) or {}
                if not (pc.get("assistant_id") or pc.get("assistantId")) or not (pc.get("api_key") or pc.get("api_key_env")):
                    raise ValueError("Target agent (vapi) missing provider_config.assistant_id or api_key")

            if not user_agent.pranthora_agent_id:
                raise ValueError("User agent missing pranthora_agent_id")

            # Ensure concurrent_calls is at least 1
            concurrent_calls = max(1, concurrent_calls)

            # Prepare request_ids: use provided list or fallback to generating
            if request_ids and len(request_ids) >= concurrent_calls:
                call_request_ids = request_ids[:concurrent_calls]
                logger.info(
                    f"Using {len(call_request_ids)} provided request_ids for concurrent calls"
                )
            else:
                # Generate request_ids if not enough provided
                if request_ids:
                    call_request_ids = request_ids[:]
                    # Generate remaining
                    for i in range(len(request_ids), concurrent_calls):
                        call_request_ids.append(str(uuid4()))
                    logger.warning(
                        f"Only {len(request_ids)} request_ids provided, generated {concurrent_calls - len(request_ids)} more"
                    )
                else:
                    # Use primary_request_id for first, generate rest
                    call_request_ids = (
                        [primary_request_id] if primary_request_id else [str(uuid4())]
                    )
                    for i in range(1, concurrent_calls):
                        call_request_ids.append(str(uuid4()))
                    logger.info(f"Generated {concurrent_calls} request_ids for concurrent calls")

            # Create concurrent conversations using provided/generated request_ids
            conversation_tasks = []
            for call_num in range(concurrent_calls):
                call_request_id = (
                    call_request_ids[call_num] if call_num < len(call_request_ids) else str(uuid4())
                )
                task = asyncio.create_task(
                    self._simulate_single_conversation(
                        test_case, target_agent, user_agent, call_num + 1, call_request_id
                    )
                )
                conversation_tasks.append(task)

            # Wait for all conversations to complete
            conversation_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)

            # Process results (no local WAV generation – recordings are handled in Pranthora)
            all_conversation_logs = []
            all_audio_data = bytearray()
            total_duration = 0
            successful_calls = 0
            failed_calls = 0
            wav_file_ids: list[str] = []
            request_ids = call_request_ids.copy()  # Start with the request_ids we used for calls

            for i, result in enumerate(conversation_results):
                if isinstance(result, Exception):
                    logger.error(f"Conversation {i + 1} failed: {result}")
                    failed_calls += 1
                    continue

                if result.get("success"):
                    successful_calls += 1
                    call_number = result.get("call_number", i + 1)
                    all_conversation_logs.extend(result.get("conversation_logs", []))
                    all_audio_data.extend(result.get("audio_data", b""))
                    total_duration = max(total_duration, result.get("duration_seconds", 0))

                    # Collect request_id for fetching transcript from Pranthora (should match call_request_ids[i])
                    req_id = result.get("request_id")
                    if req_id and req_id not in request_ids:
                        request_ids.append(req_id)
                    elif req_id and i < len(call_request_ids) and req_id != call_request_ids[i]:
                        # Log if request_id doesn't match what we expected
                        logger.warning(
                            f"Call {call_number} request_id mismatch: expected {call_request_ids[i]}, got {req_id}"
                        )
                else:
                    failed_calls += 1
                    logger.error(
                        f"Conversation {i + 1} failed: {result.get('error_message', 'Unknown error')}"
                    )

            duration_seconds = time.time() - start_time

            logger.info(
                f"All conversations completed: {successful_calls}/{concurrent_calls} successful, "
                f"{failed_calls} failed, {len(all_conversation_logs)} total turns, "
                f"{len(all_audio_data)} bytes combined audio, {duration_seconds:.2f}s total duration"
            )

            return {
                "conversation_logs": all_conversation_logs,
                "audio_data": bytes(all_audio_data),
                "combined_audio_bytes": len(all_audio_data),
                "audio_format": "mulaw 8kHz",
                "duration_seconds": duration_seconds,
                "concurrent_calls": concurrent_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "recording_file_url": None,  # Recording is handled in Pranthora, not Olympus
                "wav_file_ids": [],  # No local WAV files
                "request_ids": request_ids,  # List of request_ids for fetching transcripts from Pranthora
                "success": successful_calls > 0,
                "error_message": (
                    f"{failed_calls} out of {concurrent_calls} calls failed"
                    if failed_calls > 0
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error simulating conversations: {e}", exc_info=True)
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "combined_audio_bytes": 0,
                "error_message": str(e),
                "success": False,
            }

    async def _simulate_phone_conversation(
        self,
        test_case: TestCase,
        target_agent,
        user_agent,
        concurrent_calls: int,
    ) -> Dict[str, Any]:
        """
        Phone test execution path.

        Instead of bridging WebSockets, this uses Pranthora's /calls and /calls/end APIs
        to run real phone calls between the Pranthora agent (user agent) and the
        target phone number stored on the target agent.
        """
        from static_memory_cache import StaticMemoryCache

        try:
            logger.info(
                f"[PhoneTest] Starting phone test for case {test_case.id} with {concurrent_calls} concurrent call(s)"
            )

            # Validate user agent has a Pranthora agent id
            if not user_agent.pranthora_agent_id:
                raise ValueError("User agent missing pranthora_agent_id for phone tests")

            # Validate user agent phone_numbers config
            phone_cfg = getattr(user_agent, "phone_numbers", None) or {}
            phone_list = []
            if isinstance(phone_cfg, dict):
                raw_list = phone_cfg.get("phone_numbers") or []
                if isinstance(raw_list, list):
                    phone_list = [p for p in raw_list if isinstance(p, str) and p.strip()]

            if not phone_list:
                raise ValueError(
                    "Selected user agent has no phone_numbers configured; cannot run phone-type tests"
                )

            if concurrent_calls > len(phone_list):
                raise ValueError(
                    f"Concurrent calls ({concurrent_calls}) exceed available phone numbers "
                    f"({len(phone_list)}) for selected user agent"
                )

            # Validate target agent has phone_number in connection_metadata
            connection_metadata = getattr(target_agent, "connection_metadata", None) or {}
            target_phone = connection_metadata.get("phone_number")
            if not target_phone:
                raise ValueError(
                    "Target agent of type 'phone' must have connection_metadata.phone_number set"
                )

            # Ensure mappings between agent and phone_numbers in Pranthora
            try:
                mapping_check = await self.pranthora_client.check_agent_phone_numbers_mapings(
                    agent_id=user_agent.pranthora_agent_id,
                    phone_numbers=phone_list,
                )
                results = mapping_check.get("results", []) or []
                unmapped_numbers = [
                    r.get("phone_number")
                    for r in results
                    if not r.get("is_mapped")
                    or r.get("mapped_agent_id") != user_agent.pranthora_agent_id
                ]
                unmapped_numbers = [p for p in unmapped_numbers if p]

                if unmapped_numbers:
                    logger.info(
                        f"[PhoneTest] Mapping {len(unmapped_numbers)} phone numbers to agent {user_agent.pranthora_agent_id}"
                    )
                    await self.pranthora_client.map_agent_to_phone_number(
                        agent_id=user_agent.pranthora_agent_id,
                        phone_numbers=unmapped_numbers,
                    )
            except Exception as map_err:
                logger.error(
                    f"[PhoneTest] Failed to verify/map phone numbers for agent {user_agent.pranthora_agent_id}: {map_err}",
                    exc_info=True,
                )
                # Let execution continue; Pranthora may still route correctly if already configured

            timeout_seconds = test_case.timeout_seconds or 300

            # Initiate calls via Pranthora
            call_results: list[Dict[str, Any]] = []

            for idx in range(concurrent_calls):
                try:
                    resp = await self.pranthora_client.initiate_phone_call(
                        target_phone_number=target_phone,
                        pranthora_agent_id=user_agent.pranthora_agent_id,
                    )
                    call_results.append(resp)
                    logger.info(
                        f"[PhoneTest] Started call {idx + 1}/{concurrent_calls}: call_sid={resp.get('call_sid')}, "
                        f"from={resp.get('from_phone_number')}, request_id={resp.get('request_id')}"
                    )
                except Exception as e:
                    logger.error(f"[PhoneTest] Failed to initiate phone call {idx + 1}: {e}", exc_info=True)

            if not call_results:
                return {
                    "conversation_logs": [],
                    "audio_data": b"",
                    "combined_audio_bytes": 0,
                    "error_message": "Failed to initiate any phone calls via Pranthora",
                    "success": False,
                }

            # Schedule call termination after timeout for each call
            async def _end_call_later(call_sid: str, from_phone: str):
                try:
                    await asyncio.sleep(timeout_seconds)
                    await self.pranthora_client.end_phone_call(call_sid, from_phone)
                    # Give Pranthora a small buffer to persist transcripts/recordings
                    await asyncio.sleep(5)
                    logger.info(
                        f"[PhoneTest] Ended phone call call_sid={call_sid}, from={from_phone} after timeout={timeout_seconds}s"
                    )
                except Exception as e:
                    logger.error(
                        f"[PhoneTest] Error ending phone call call_sid={call_sid}, from={from_phone}: {e}",
                        exc_info=True,
                    )

            end_tasks = []
            for r in call_results:
                call_sid = r.get("call_sid")
                from_phone = r.get("from_phone_number")
                if call_sid and from_phone:
                    end_tasks.append(asyncio.create_task(_end_call_later(call_sid, from_phone)))

            # Wait for all calls to be ended and recordings/transcripts to be persisted
            if end_tasks:
                await asyncio.gather(*end_tasks, return_exceptions=True)

            # Collect request_ids for transcript fetching
            request_ids: list[str] = []
            for r in call_results:
                rid = r.get("request_id") or r.get("session_id")
                if isinstance(rid, str) and rid:
                    request_ids.append(rid)

            logger.info(
                f"[PhoneTest] Initiated {len(call_results)} phone calls for test case {test_case.id}. "
                f"Timeout={timeout_seconds}s, request_ids={request_ids}"
            )

            return {
                "conversation_logs": [],
                "audio_data": b"",
                "combined_audio_bytes": 0,
                "audio_format": "external_pranthora_recording",
                "duration_seconds": timeout_seconds,
                "concurrent_calls": concurrent_calls,
                "successful_calls": len(call_results),
                "failed_calls": 0,
                "recording_file_url": None,
                "wav_file_ids": [],
                "request_ids": request_ids,
                "success": len(call_results) > 0,
                "error_message": None,
            }
        except Exception as e:
            logger.error(f"[PhoneTest] Error during phone test execution: {e}", exc_info=True)
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "combined_audio_bytes": 0,
                "error_message": str(e),
                "success": False,
            }

    async def _simulate_single_conversation(
        self,
        test_case: TestCase,
        target_agent,
        user_agent,
        call_number: int,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a single conversation between target and user agents.
        Used internally for concurrent call support.
        """
        try:
            conversation_logs = []
            combined_audio_data = bytearray()
            conv_start_time = time.time()

            # Generate unique request_id for this call if not provided
            if not request_id:
                request_id = str(uuid4())

            logger.info(
                f"[Call {call_number}] Starting conversation simulation with request_id: {request_id}"
            )

            # Use Pranthora base URL from config to construct websocket URLs
            from static_memory_cache import StaticMemoryCache

            pranthora_base_url = StaticMemoryCache.get_pranthora_base_url()
            if not pranthora_base_url:
                raise ValueError("Pranthora base URL not configured")

            # Convert HTTP to WS
            if pranthora_base_url.startswith("https://"):
                base_ws_url = pranthora_base_url.replace("https://", "wss://")
            else:
                base_ws_url = pranthora_base_url.replace("http://", "ws://")

            user_ws_url = (
                f"{base_ws_url}/api/call/media-stream/agents/{user_agent.pranthora_agent_id}"
            )

            # Generate unique call SIDs for this conversation
            call_sid_target = str(uuid.uuid4())
            call_sid_user = request_id  # Use request_id as call_sid for User Agent to ensure transcript matching
            user_ws_url = f"{user_ws_url}?call_sid={call_sid_user}"

            # Queues for this conversation
            target_to_user_queue = asyncio.Queue()
            user_to_target_queue = asyncio.Queue()
            stop_event = asyncio.Event()

            # Isolated PCM recording buffer for this specific call
            # Pre-allocate with estimated size (5 min call @ 8kHz = ~4.8MB PCM)
            isolated_pcm_frames = bytearray()

            # Async recording queue - decouples recording from audio streaming
            # maxsize=1000 = ~20 sec buffer; if exceeded, something is catastrophically wrong
            recording_queue = asyncio.Queue(maxsize=1000)
            chunks_recorded = 0

            def record_audio_bridge(audio_bytes: bytes, source: str):
                """Non-blocking: queue audio for background processing."""
                try:
                    recording_queue.put_nowait((audio_bytes, source))
                except asyncio.QueueFull:
                    logger.warning(
                        f"[Call {call_number}] Recording queue full (1000 chunks), dropping chunk"
                    )

            async def process_recording_queue():
                """Background task: processes queued audio without blocking main loop."""
                nonlocal chunks_recorded
                while not stop_event.is_set() or not recording_queue.empty():
                    try:
                        audio_bytes, source = await asyncio.wait_for(
                            recording_queue.get(), timeout=0.1
                        )
                        # Do the actual recording work here (off the critical audio path)
                        combined_audio_data.extend(audio_bytes)
                        pcm = audioop.ulaw2lin(audio_bytes, 2)  # 16-bit PCM
                        isolated_pcm_frames.extend(pcm)
                        chunks_recorded += 1
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"[Call {call_number}] Recording processor error: {e}")

            # Log conversation start
            conversation_logs.append(
                {
                    "call_number": call_number,
                    "turn": 0,
                    "type": "system",
                    "content": f"Starting conversation test with {len(test_case.goals)} goals",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            # Synchronization: both connections must be ready before write loops start
            target_ready = asyncio.Event()
            user_ready = asyncio.Event()

            # Target Agent Connection (custom WebSocket/HTTP or Vapi via factory)
            target_manager = await create_target_connection(
                target_agent,
                call_sid=call_sid_target,
                incoming_queue=user_to_target_queue,
                outgoing_queue=target_to_user_queue,
                stop_event=stop_event,
                my_ready=target_ready,
                other_ready=user_ready,
                record_sent_callback=lambda audio: record_audio_bridge(audio, "user_to_target"),
                sync_timeout=self.connection_sync_timeout,
                pranthora_base_url=pranthora_base_url,
            )
            target_task = asyncio.create_task(target_manager.connect())

            # User Agent Connection
            user_manager = AgentConnectionManager(
                name="User",
                ws_url=user_ws_url,
                call_sid=call_sid_user,
                incoming_queue=target_to_user_queue,  # Audio IN to this agent (read from other agent's output)
                outgoing_queue=user_to_target_queue,  # Audio OUT from this agent (read from ws, put here)
                stop_event=stop_event,
                my_ready=user_ready,
                other_ready=target_ready,
                sync_timeout=self.connection_sync_timeout,
                extra_headers={"x-pranthora-callid": request_id} if request_id else None,
                record_sent_callback=lambda audio: record_audio_bridge(audio, "target_to_user"),
            )
            user_task = asyncio.create_task(user_manager.connect())

            # Start the recording processor as a background task
            recording_task = asyncio.create_task(process_recording_queue())

            # Wait for conversation to complete or timeout
            timeout_seconds = test_case.timeout_seconds or 300
            connection_failed = False
            error_message = None

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(target_task, user_task, return_exceptions=True),
                    timeout=timeout_seconds,
                )

                # Check if any of the tasks failed
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        connection_failed = True
                        task_name = "target" if i == 0 else "user"
                        error_message = f"{task_name} agent connection failed: {result}"
                        logger.error(f"[Call {call_number}] {error_message}")
                        break

            except asyncio.TimeoutError:
                logger.info(
                    f"[Call {call_number}] Conversation timeout reached ({timeout_seconds}s)"
                )
                conversation_logs.append(
                    {
                        "call_number": call_number,
                        "turn": len(conversation_logs),
                        "type": "system",
                        "content": f"Conversation timeout reached after {timeout_seconds} seconds",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                # Timeout means conversation completed (just took too long), not failed
                # Set error_message to indicate timeout but continue with normal processing
                error_message = f"Conversation timeout after {timeout_seconds} seconds"
            finally:
                # Stop connections
                stop_event.set()
                target_task.cancel()
                user_task.cancel()
                await asyncio.gather(target_task, user_task, return_exceptions=True)
                # Wait for recording queue to drain (processor will exit when queue empty and stop_event set)
                try:
                    await asyncio.wait_for(recording_task, timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[Call {call_number}] Recording processor didn't finish in time, cancelling"
                    )
                    recording_task.cancel()
                    await asyncio.gather(recording_task, return_exceptions=True)

            # If connection failed, return failure immediately
            if connection_failed:
                return {
                    "conversation_logs": conversation_logs,
                    "audio_data": b"",
                    "pcm_frames": b"",  # Include pcm_frames for consistency
                    "combined_audio_bytes": 0,
                    "error_message": error_message,
                    "call_number": call_number,
                    "success": False,
                }

            conv_duration = time.time() - conv_start_time
            audio_data = bytes(combined_audio_data)

            logger.info(
                f"[Call {call_number}] Conversation completed: {chunks_recorded} chunks recorded, "
                f"{len(audio_data)} bytes audio, {len(isolated_pcm_frames)} PCM bytes, {conv_duration:.2f}s"
            )

            return {
                "conversation_logs": conversation_logs,
                "audio_data": audio_data,
                "pcm_frames": bytes(
                    isolated_pcm_frames
                ),  # Return PCM frames for individual WAV creation
                "combined_audio_bytes": len(audio_data),
                "audio_format": "mulaw 8kHz",
                "duration_seconds": conv_duration,
                "call_number": call_number,
                "request_id": request_id,  # Store request_id for fetching transcript from Pranthora
                "error_message": error_message if "error_message" in locals() else None,
                "success": True,
            }

        except Exception as e:
            logger.error(f"[Call {call_number}] Error simulating conversation: {e}", exc_info=True)
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "pcm_frames": b"",  # Include pcm_frames for consistency
                "combined_audio_bytes": 0,
                "error_message": str(e),
                "call_number": call_number,
                "success": False,
            }

    # _connect_target_agent and _connect_user_agent removed in favor of AgentConnectionManager

    async def _evaluate_test_results_sequential(
        self,
        test_case: TestCase,
        conversation_result: Dict[str, Any],
        user_agent,
        test_run_id: UUID,
        result_id: UUID,
    ):
        """Sequential evaluation with retry logic for fetching transcripts."""
        try:
            request_ids = self._extract_request_ids(conversation_result)
            if not request_ids:
                logger.warning(f"No request_ids found for test case {test_case.id}")
                # Update status to failed if no request_ids
                await self.test_result_service.update(
                    result_id,
                    {
                        "status": "failed",
                        "error_message": "No request_ids found for transcript fetching",
                    },
                )
                return None

            # Fetch transcript with more retries (10 retries, 5 second delay = up to 50 seconds wait)
            transcript = await self._fetch_transcript_with_retry(
                request_ids, max_retries=10, retry_delay=5
            )

            if not transcript:
                # All retries exhausted, update status to failed
                logger.error(
                    f"Failed to fetch transcript after all retries for test case {test_case.id}"
                )
                await self.test_result_service.update(
                    result_id,
                    {
                        "status": "failed",
                        "error_message": "Failed to fetch transcript after all retries",
                    },
                )
                return None

            # Run evaluation
            evaluation_result = await self._run_evaluation(test_case, transcript, user_agent)

            # Update evaluation result
            await self._update_test_case_result_with_evaluation(
                test_run_id, test_case.id, evaluation_result
            )

            # Determine final status based on evaluation
            final_status = self._determine_test_status(evaluation_result)

            # Update status to complete (pass/alert/fail based on evaluation)
            await self.test_result_service.update(
                result_id, {"status": final_status, "evaluation_result": evaluation_result}
            )

            logger.info(
                f"Evaluation completed for test case {test_case.id} with status: {final_status}"
            )
            return evaluation_result

        except Exception as e:
            logger.error(f"Sequential evaluation error: {e}", exc_info=True)
            # Update status to failed on exception
            try:
                await self.test_result_service.update(
                    result_id, {"status": "failed", "error_message": f"Evaluation error: {str(e)}"}
                )
            except Exception as update_error:
                logger.error(f"Failed to update status after evaluation error: {update_error}")
            return None

    def _extract_request_ids(self, conversation_result: Dict[str, Any]) -> List[str]:
        """Extract request_ids from conversation result."""
        request_ids = conversation_result.get("request_ids", [])
        if not request_ids:
            single_id = conversation_result.get("request_id")
            if single_id:
                request_ids = [single_id]
        return request_ids

    async def _fetch_transcript_with_retry(
        self, request_ids: List[str], max_retries: int = 10, retry_delay: int = 5
    ) -> List[Dict[str, Any]]:
        """Fetch transcript with retry logic. Returns empty list if all retries exhausted."""
        transcript = []
        for request_id in request_ids:
            fetched = False
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(
                            f"Retry {attempt}/{max_retries} for request_id {request_id} (waiting {retry_delay}s)"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.info(f"Fetching transcript for request_id: {request_id}")

                    call_logs = await self.pranthora_client.get_call_logs(request_id)
                    if call_logs and call_logs.get("call_transcript"):
                        transcript.extend(self._parse_transcript(call_logs["call_transcript"]))
                        logger.info(
                            f"✅ Successfully fetched {len(call_logs['call_transcript'])} messages for request_id: {request_id}"
                        )
                        fetched = True
                        break
                    elif call_logs:
                        logger.warning(
                            f"Call logs found for {request_id} but no transcript available"
                        )
                    else:
                        logger.debug(
                            f"Call logs not found for request_id {request_id} (attempt {attempt + 1}/{max_retries + 1})"
                        )

                except Exception as e:
                    if attempt < max_retries:
                        logger.warning(
                            f"Error fetching transcript for {request_id} (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                    else:
                        logger.error(
                            f"Failed to fetch transcript for {request_id} after {max_retries + 1} attempts: {e}"
                        )

            if not fetched:
                logger.error(
                    f"❌ Failed to fetch transcript for request_id {request_id} after all {max_retries + 1} attempts"
                )

        return transcript

    def _parse_transcript(self, call_transcript: List[Any]) -> List[Dict[str, Any]]:
        """Parse transcript messages to standard format."""
        parsed = []
        for msg in call_transcript:
            if isinstance(msg, dict):
                role = self._normalize_role(msg.get("role", "unknown"))
                content = msg.get("content", msg.get("message", ""))
                if content:
                    parsed.append(
                        TranscriptMessage(
                            role=role, content=content, timestamp=msg.get("timestamp", "")
                        ).model_dump()
                    )
        return parsed

    def _normalize_role(self, role: Any) -> str:
        """Normalize role to user or assistant."""
        if hasattr(role, "value"):
            role = role.value
        elif hasattr(role, "name"):
            role = role.name.lower()
        role_str = str(role).lower()
        return "user" if role_str == "user" else "assistant"

    async def _run_evaluation(
        self, test_case: TestCase, transcript: List[Dict[str, Any]], user_agent
    ) -> Dict[str, Any]:
        """Run evaluation with transcript."""
        try:
            return await self.evaluation_service.evaluate_with_retry(
                transcript=transcript,
                goals=test_case.goals or [],
                evaluation_criteria=test_case.evaluation_criteria or [],
                test_case_name=test_case.name,
                max_retries=2,
            )
        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)
            return {
                "error": str(e),
                "overall_score": 0.0,
                "overall_status": "failed",
                "summary": f"Evaluation failed: {str(e)}",
                "timestamp": time.time(),
            }

    async def _update_test_case_result_with_evaluation(
        self, test_run_id: UUID, test_case_id: UUID, evaluation_result: Dict[str, Any]
    ):
        """Update test case result with evaluation."""
        try:
            existing = await self.test_result_service.get_result_by_test_run_and_case(
                test_run_id, test_case_id
            )
            if existing:
                await self.test_result_service.update(
                    existing.id, {"evaluation_result": evaluation_result}
                )
                logger.info(f"Updated evaluation for test case {test_case_id}")
        except Exception as e:
            logger.error(f"Failed to update evaluation: {e}", exc_info=True)

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
        self, test_run_id: UUID, status: str, passed_count: int, failed_count: int, alert_count: int
    ):
        """Update the test run status and statistics."""
        try:
            update_data = {
                "status": status,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "alert_count": alert_count,
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
