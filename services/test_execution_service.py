"""
Test execution service for running test cases and test suites.

This module provides functionality to execute individual test cases or entire test suites,
simulate conversations, evaluate results, and store execution history.
"""

import asyncio
import json
import base64
import time
import uuid
import wave
import audioop
import io
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from services.database_service import DatabaseService
from services.test_case_service import TestCaseService
from services.test_suite_service import TestSuiteService
from services.target_agent_service import TargetAgentService
from services.user_agent_service import UserAgentService
from services.test_history_service import TestRunHistoryService, TestCaseResultService
from services.recording_storage_service import RecordingStorageService
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
        self.recording_service = RecordingStorageService()
        self.pranthora_client = PranthoraApiClient()

        # Recording setup
        self.sample_rate = 8000  # μ-law sample rate



    async def run_test_suite(
        self,
        test_suite_id: UUID,
        user_id: UUID,
        concurrent_calls: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> UUID:
        """
        Run all active test cases in a test suite.

        Args:
            test_suite_id: ID of the test suite to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)
            request_id: Request ID from x-pranthora-callid header (used as primary key)

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
            test_run_id = await self._create_test_run(test_suite_id, user_id, len(test_cases), request_id)

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
        concurrent_calls: Optional[int] = None,
        request_id: Optional[str] = None
    ) -> UUID:
        """
        Run a single test case.

        Args:
            test_case_id: ID of the test case to run
            user_id: ID of the user running the test
            concurrent_calls: Number of concurrent calls (overrides default)
            request_id: Request ID from x-pranthora-callid header (used as primary key)

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
            test_run_id = await self._create_test_run(test_case.test_suite_id, user_id, 1, request_id)

            # Execute single test case asynchronously
            asyncio.create_task(
                self._execute_single_test_case_async(test_run_id, test_case, concurrent_calls)
            )

            logger.info(f"Started single test case execution: {test_run_id}")
            return test_run_id

        except Exception as e:
            logger.error(f"Error starting single test case execution: {e}")
            raise

    async def _create_test_run(self, test_suite_id: UUID, user_id: UUID, total_cases: int, request_id: Optional[str] = None) -> UUID:
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
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

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

            logger.info(f"Completed test run {test_run_id}: status={final_status}, {passed_count} passed, {failed_count} failed, {alert_count} alerts")

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

            # Status values: completed, failed
            passed_count = 1 if result["status"] == "completed" else 0
            failed_count = 1 if result["status"] == "failed" else 0
            alert_count = 0

            # For single test case execution, update test run status based on result
            test_run_status = "failed" if result["status"] == "failed" else "completed"
            await self._update_test_run_status(
                test_run_id, test_run_status, passed_count, failed_count, alert_count
            )

            logger.info(f"Completed single test case execution {test_run_id}: {result['status']}, run status: {test_run_status}")

        except Exception as e:
            logger.error(f"Error in single test case execution: {e}")
            # Don't update test run status on error either - let external processes handle completion

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
                logger.warning(f"Target agent not found for test suite {test_case.test_suite_id}")
                # For development/testing, allow execution without target agent but fail gracefully
                return {
                    "result_id": None,
                    "status": "failed",
                    "error": f"Target agent not found for test suite {test_case.test_suite_id}"
                }

            if not user_agent:
                logger.warning(f"User agent not found for test suite {test_case.test_suite_id}")
                # For development/testing, allow execution without user agent but fail gracefully
                return {
                    "result_id": None,
                    "status": "failed",
                    "error": f"User agent not found for test suite {test_case.test_suite_id}"
                }

            # Create initial test case result with running status
            # Recording URL will be added when test completes
            initial_result_data = {
                "test_run_id": str(test_run_id),
                "test_case_id": str(test_case.id),
                "test_suite_id": str(test_case.test_suite_id),
                "status": "running",
                "conversation_logs": [],
                "evaluation_result": None,
                "error_message": None
            }
            initial_result_id = await self.test_result_service.create(initial_result_data)
            logger.info(f"Created test case result {initial_result_id} with initial running status for test case {test_case.id}")

            # Simulate conversation using goals/prompts
            conversation_result = await self._simulate_conversation(
                test_case, target_agent, user_agent, concurrent_calls or test_case.default_concurrent_calls, test_run_id
            )

            # Determine status based on conversation result
            if conversation_result.get("success", False):
                # WebSocket connections successful - conversation completed
                if conversation_result.get("error_message") and "timeout" in conversation_result.get("error_message", "").lower():
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
                logger.error(f"Conversation failed for test case {test_case.id}: {conversation_result.get('error_message', 'Unknown error')}")

            # Check if test case result already exists for this test run and test case
            existing_result = await self.test_result_service.get_result_by_test_run_and_case(test_run_id, test_case.id)

            if existing_result:
                # Update existing test case result with final status and recording URL
                update_data = {
                    "status": status,
                    "conversation_logs": conversation_result.get("conversation_logs", []),
                    "error_message": conversation_result.get("error_message")
                }

                # Add recording URL if available (from combined recording at suite level)
                if conversation_result.get("recording_file_url"):
                    update_data["recording_file_url"] = conversation_result["recording_file_url"]

                success = await self.test_result_service.update(existing_result.id, update_data)
                result_id = existing_result.id
                logger.info(f"Updated existing test case result {result_id} with status {status}")
            else:
                # Create new test case result (fallback)
                result_data = {
                    "test_run_id": str(test_run_id),
                    "test_case_id": str(test_case.id),
                    "test_suite_id": str(test_case.test_suite_id),
                    "status": status,
                    "conversation_logs": conversation_result.get("conversation_logs", []),
                    "evaluation_result": None,
                    "error_message": conversation_result.get("error_message")
                }

                # Add recording URL if available
                if conversation_result.get("recording_file_url"):
                    result_data["recording_file_url"] = conversation_result["recording_file_url"]

                result_id = await self.test_result_service.create(result_data)
                logger.info(f"Created new test case result {result_id} with status {status}")

            return {
                "result_id": result_id,
                "status": status,
                "conversation_result": conversation_result
            }

        except Exception as e:
            logger.error(f"Error executing test case {test_case.id}: {e}")
            # Update or create failed result
            try:
                # Check if test case result already exists
                existing_result = await self.test_result_service.get_result_by_test_run_and_case(test_run_id, test_case.id)

                if existing_result:
                    # Update existing result to failed
                    await self.test_result_service.update(
                        existing_result.id,
                        {"status": "failed", "error_message": str(e)}
                    )
                    result_id = existing_result.id
                else:
                    # Create new failed result
                    result_data = {
                        "test_run_id": str(test_run_id),
                        "test_case_id": str(test_case.id),
                        "test_suite_id": str(test_case.test_suite_id),
                        "status": "failed",
                        "error_message": str(e)
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
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate conversations by connecting to user agent and target agent websockets,
        bridging audio between them, and recording the conversation(s).

        Supports concurrent calls - creates multiple simultaneous conversations when concurrent_calls > 1.
        """
        try:
            start_time = time.time()

            logger.info(f"Starting conversation simulation for test case {test_case.id} with {concurrent_calls} concurrent call(s)")
            logger.info(f"Target agent: {target_agent.websocket_url}, User agent: {user_agent.pranthora_agent_id}")

            # Validate agents have required information
            if not target_agent.websocket_url:
                raise ValueError("Target agent missing websocket_url")

            if not user_agent.pranthora_agent_id:
                raise ValueError("User agent missing pranthora_agent_id")

            # Ensure concurrent_calls is at least 1
            concurrent_calls = max(1, concurrent_calls)

            # Create concurrent conversations
            conversation_tasks = []
            for call_num in range(concurrent_calls):
                task = asyncio.create_task(
                    self._simulate_single_conversation(
                        test_case, target_agent, user_agent, call_num + 1, request_id
                    )
                )
                conversation_tasks.append(task)

            # Wait for all conversations to complete
            conversation_results = await asyncio.gather(*conversation_tasks, return_exceptions=True)

            # Process results
            all_conversation_logs = []
            all_audio_data = bytearray()
            total_duration = 0
            successful_calls = 0
            failed_calls = 0

            for i, result in enumerate(conversation_results):
                if isinstance(result, Exception):
                    logger.error(f"Conversation {i+1} failed: {result}")
                    failed_calls += 1
                    continue

                if result.get("success"):
                    successful_calls += 1
                    all_conversation_logs.extend(result.get("conversation_logs", []))
                    all_audio_data.extend(result.get("audio_data", b""))
                    total_duration = max(total_duration, result.get("duration_seconds", 0))
                else:
                    failed_calls += 1
                    logger.error(f"Conversation {i+1} failed: {result.get('error_message', 'Unknown error')}")

            duration_seconds = time.time() - start_time

            # Create combined WAV file from all conversations
            recording_file_url = None
            if all_audio_data:
                try:
                    # Convert combined μ-law audio to PCM
                    combined_pcm = audioop.ulaw2lin(bytes(all_audio_data), 2)  # 16-bit PCM

                    # Create WAV file data in memory
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(1)      # Mono
                        wf.setsampwidth(2)      # 16-bit PCM
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(combined_pcm)

                    wav_data = wav_buffer.getvalue()
                    wav_filename = f"test_case_{test_case.id}_call_1_recording.wav"

                    # Upload combined WAV file to Supabase
                    wav_file_id = await self.recording_service.upload_recording_file(
                        file_content=wav_data,
                        file_name=wav_filename,
                        content_type="audio/wav"
                    )

                    if wav_file_id:
                        # Generate signed URL immediately after upload
                        from data_layer.supabase_client import get_supabase_client
                        supabase_client = await get_supabase_client()
                        file_path = f"{wav_file_id}_{wav_filename}"
                        recording_file_url = await supabase_client.create_signed_url("recording_files", file_path, 3600)

                        if recording_file_url:
                            logger.info(f"📤 Combined WAV file uploaded and URL generated: {file_path}")
                        else:
                            logger.error("Failed to generate signed URL for combined WAV file")
                    else:
                        logger.error("Failed to upload combined WAV file")

                except Exception as e:
                    logger.error(f"Failed to create/upload combined WAV file: {e}")

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
                "recording_file_url": recording_file_url,  # Signed URL for the combined recording
                "success": successful_calls > 0,
                "error_message": f"{failed_calls} out of {concurrent_calls} calls failed" if failed_calls > 0 else None
            }

        except Exception as e:
            logger.error(f"Error simulating conversations: {e}", exc_info=True)
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "combined_audio_bytes": 0,
                "error_message": str(e),
                "success": False
            }

    async def _simulate_single_conversation(
        self,
        test_case: TestCase,
        target_agent,
        user_agent,
        call_number: int,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate a single conversation between target and user agents.
        Used internally for concurrent call support.
        """
        try:
            conversation_logs = []
            combined_audio_data = bytearray()
            conv_start_time = time.time()

            logger.info(f"[Call {call_number}] Starting conversation simulation")

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

            # For target agent, we still need a websocket URL. If target_agent has a websocket_url, use it,
            # otherwise construct one. But for now, let's use the configured URL and replace the port
            target_ws_url = target_agent.websocket_url
            if not target_ws_url.startswith(("ws://", "wss://")):
                raise ValueError(f"Invalid target agent websocket URL: {target_ws_url}")

            # Replace the port in target_ws_url with the port from pranthora_base_url
            import re
            port_match = re.search(r':(\d+)', pranthora_base_url)
            if port_match:
                pranthora_port = port_match.group(1)
                target_ws_url = re.sub(r':\d+', f':{pranthora_port}', target_ws_url)

            user_ws_url = f"{base_ws_url}/api/call/media-stream/agents/{user_agent.pranthora_agent_id}"

            # Generate unique call SIDs for this conversation
            call_sid_target = str(uuid.uuid4())
            call_sid_user = str(uuid.uuid4())

            # Add call_sid to URLs
            if "call_sid=" not in target_ws_url:
                separator = "&" if "?" in target_ws_url else "?"
                target_ws_url = f"{target_ws_url}{separator}call_sid={call_sid_target}"
            user_ws_url = f"{user_ws_url}?call_sid={call_sid_user}"

            # Queues for this conversation
            target_to_user_queue = asyncio.Queue()
            user_to_target_queue = asyncio.Queue()
            stop_event = asyncio.Event()

            # Isolated PCM recording buffer for this specific call
            isolated_pcm_frames = bytearray()

            # Audio recording for this conversation only
            def record_audio_bridge(audio_bytes: bytes, source: str):
                """Record audio from bridge for this isolated conversation."""
                try:
                    combined_audio_data.extend(audio_bytes)

                    # Convert μ-law to PCM for this call's isolated WAV recording
                    pcm = audioop.ulaw2lin(audio_bytes, 2)  # 16-bit PCM
                    isolated_pcm_frames.extend(pcm)

                    conversation_logs.append({
                        "call_number": call_number,
                        "turn": len(conversation_logs),
                        "type": f"{source}_audio",
                        "content": f"Audio received from {source} agent ({len(audio_bytes)} bytes)",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f"[Call {call_number}] Bridge recording error from {source}: {e}")

            # Log conversation start
            conversation_logs.append({
                "call_number": call_number,
                "turn": 0,
                "type": "system",
                "content": f"Starting conversation test with {len(test_case.goals)} goals",
                "timestamp": datetime.utcnow().isoformat()
            })

            # Start websocket connections for this conversation
            target_task = asyncio.create_task(
                self._connect_target_agent(
                    target_ws_url,
                    call_sid_target,
                    target_to_user_queue,
                    user_to_target_queue,
                    stop_event,
                    lambda audio: record_audio_bridge(audio, "target"),
                    conversation_logs
                )
            )

            user_task = asyncio.create_task(
                self._connect_user_agent(
                    user_ws_url,
                    call_sid_user,
                    target_to_user_queue,
                    user_to_target_queue,
                    stop_event,
                    lambda audio: record_audio_bridge(audio, "user"),
                    conversation_logs,
                    request_id
                )
            )

            # Wait for conversation to complete or timeout
            timeout_seconds = test_case.timeout_seconds or 300
            connection_failed = False
            error_message = None

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(target_task, user_task, return_exceptions=True),
                    timeout=timeout_seconds
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
                logger.info(f"[Call {call_number}] Conversation timeout reached ({timeout_seconds}s)")
                conversation_logs.append({
                    "call_number": call_number,
                    "turn": len(conversation_logs),
                    "type": "system",
                    "content": f"Conversation timeout reached after {timeout_seconds} seconds",
                    "timestamp": datetime.utcnow().isoformat()
                })
                # Timeout means conversation completed (just took too long), not failed
                # Set error_message to indicate timeout but continue with normal processing
                error_message = f"Conversation timeout after {timeout_seconds} seconds"
            finally:
                # Stop connections
                stop_event.set()
                target_task.cancel()
                user_task.cancel()
                await asyncio.gather(target_task, user_task, return_exceptions=True)

            # If connection failed, return failure immediately
            if connection_failed:
                return {
                    "conversation_logs": conversation_logs,
                    "audio_data": b"",
                    "combined_audio_bytes": 0,
                    "error_message": error_message,
                    "success": False
                }

            conv_duration = time.time() - conv_start_time
            audio_data = bytes(combined_audio_data)

            # Don't create individual WAV files - only combined file will be created
            # in the parent _simulate_conversation function

            logger.info(
                f"[Call {call_number}] Isolated conversation completed: {len(conversation_logs)} turns, "
                f"{len(audio_data)} bytes audio, {len(isolated_pcm_frames)} PCM bytes, {conv_duration:.2f}s duration"
            )

            return {
                "conversation_logs": conversation_logs,
                "audio_data": audio_data,
                "combined_audio_bytes": len(audio_data),
                "audio_format": "mulaw 8kHz",
                "duration_seconds": conv_duration,
                "call_number": call_number,
                "error_message": error_message if 'error_message' in locals() else None,
                "success": True
            }

        except Exception as e:
            logger.error(f"[Call {call_number}] Error simulating conversation: {e}", exc_info=True)
            return {
                "conversation_logs": [],
                "audio_data": b"",
                "combined_audio_bytes": 0,
                "error_message": str(e),
                "success": False
            }

    async def _connect_target_agent(
        self,
        ws_url: str,
        call_sid: str,
        outgoing_queue: asyncio.Queue,
        incoming_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        record_callback: callable,
        conversation_logs: List[Dict[str, Any]]
    ):
        """Connect to target agent websocket (media-stream endpoint)."""
        logger.info(f"[Target] Connecting to {ws_url}")

        try:
            async with websockets.connect(ws_url) as websocket:
                # Send start event
                start_event = {
                    "event": "start",
                    "sequenceNumber": "1",
                    "start": {
                        "accountSid": "AC_SIMULATION",
                        "callSid": call_sid,
                        "streamSid": f"stream_{call_sid}",
                        "tracks": ["inbound"],
                        "customParameters": {}
                    },
                    "streamSid": f"stream_{call_sid}"
                }
                await websocket.send(json.dumps(start_event))
                logger.info(f"[Target] Sent start event")

                # Read from target agent
                async def read_from_ws():
                    try:
                        async for message in websocket:
                            if stop_event.is_set():
                                break

                            data = json.loads(message)
                            event_type = data.get("event")

                            if event_type == "media":
                                payload = data["media"]["payload"]
                                audio_bytes = base64.b64decode(payload)

                                # Record audio from target agent
                                record_callback(audio_bytes)

                                # Send to user agent
                                await outgoing_queue.put(audio_bytes)

                            elif event_type == "mark":
                                logger.info(f"[Target] Mark: {data.get('mark', {}).get('name')}")
                            elif event_type == "clear":
                                logger.info(f"[Target] Clear received")
                                # Clear incoming queue
                                while not incoming_queue.empty():
                                    try:
                                        incoming_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                            elif event_type == "stop":
                                logger.info(f"[Target] Stop received")
                                break
                    except Exception as e:
                        logger.error(f"[Target] Read error: {e}")

                # Write to target agent (20ms cadence)
                async def write_to_ws():
                    stream_sid = f"stream_{call_sid}"
                    next_tick = time.time()

                    try:
                        while not stop_event.is_set():
                            # Maintain 20ms cadence
                            next_tick += 0.02
                            sleep_time = next_tick - time.time()
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)

                            payload_to_send = None

                            # Check for audio from user agent
                            if not incoming_queue.empty():
                                try:
                                    audio_data = incoming_queue.get_nowait()
                                    payload_to_send = base64.b64encode(audio_data).decode('utf-8')
                                except asyncio.QueueEmpty:
                                    pass

                            # Send silence if no audio
                            if not payload_to_send:
                                silence = b"\xff" * 160  # μ-law silence
                                payload_to_send = base64.b64encode(silence).decode('utf-8')

                            # Send media event
                            media_event = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": payload_to_send
                                }
                            }
                            await websocket.send(json.dumps(media_event))

                    except asyncio.CancelledError:
                        pass
                    except (ConnectionClosedOK, ConnectionClosedError):
                        # Normal websocket closure or clean disconnect - not an error
                        logger.debug(f"[Target] WebSocket closed normally")
                        pass
                    except Exception as e:
                        logger.error(f"[Target] Write error: {e}")

                reader_task = asyncio.create_task(read_from_ws())
                writer_task = asyncio.create_task(write_to_ws())

                await reader_task
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[Target] Connection failed: {e}")
            raise

    async def _connect_user_agent(
        self,
        ws_url: str,
        call_sid: str,
        incoming_queue: asyncio.Queue,
        outgoing_queue: asyncio.Queue,
        stop_event: asyncio.Event,
        record_callback: callable,
        conversation_logs: List[Dict[str, Any]],
        request_id: Optional[str] = None
    ):
        """Connect to user agent websocket (media-stream endpoint)."""
        logger.info(f"[User] Connecting to {ws_url}")

        # Add request_id to headers if provided
        additional_headers = {}
        if request_id:
            additional_headers["x-pranthora-callid"] = request_id
            logger.info(f"[User] Request ID: {request_id}")

        try:
            async with websockets.connect(ws_url, additional_headers=additional_headers) as websocket:
                # Send start event (Twilio-style)
                start_event = {
                    "event": "start",
                    "sequenceNumber": "1",
                    "start": {
                        "accountSid": "AC_SIMULATION",
                        "callSid": call_sid,
                        "streamSid": f"stream_{call_sid}",
                        "tracks": ["inbound"],
                        "customParameters": {},
                    },
                    "streamSid": f"stream_{call_sid}",
                }
                await websocket.send(json.dumps(start_event))
                logger.info(f"[User] Sent start event")

                # Read from user agent (JSON messages)
                async def read_from_ws():
                    try:
                        async for message in websocket:
                            if stop_event.is_set():
                                break

                            data = json.loads(message)
                            event_type = data.get("event")

                            if event_type == "media":
                                payload = data["media"]["payload"]
                                audio_bytes = base64.b64decode(payload)

                                # Record audio from user agent
                                record_callback(audio_bytes)

                                # Send to target agent
                                await outgoing_queue.put(audio_bytes)

                            elif event_type == "mark":
                                logger.info(f"[User] Mark: {data.get('mark', {}).get('name')}")
                            elif event_type == "clear":
                                logger.info(f"[User] Clear received")
                                # Clear incoming queue
                                while not incoming_queue.empty():
                                    try:
                                        incoming_queue.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                            elif event_type == "stop":
                                logger.info(f"[User] Stop received")
                                break
                    except Exception as e:
                        logger.error(f"[User] Read error: {e}")

                # Write to user agent (20ms cadence, JSON media events)
                async def write_to_ws():
                    stream_sid = f"stream_{call_sid}"
                    next_tick = time.time()

                    try:
                        while not stop_event.is_set():
                            # Maintain 20ms cadence
                            next_tick += 0.02
                            sleep_time = next_tick - time.time()
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)

                            payload_to_send = None

                            # Check for audio from target agent
                            if not incoming_queue.empty():
                                try:
                                    audio_data = incoming_queue.get_nowait()
                                    payload_to_send = base64.b64encode(audio_data).decode('utf-8')
                                except asyncio.QueueEmpty:
                                    pass

                            # Send silence if no audio
                            if not payload_to_send:
                                silence = b"\xff" * 160  # μ-law silence
                                payload_to_send = base64.b64encode(silence).decode('utf-8')

                            # Send media event
                            media_event = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": payload_to_send
                                }
                            }
                            await websocket.send(json.dumps(media_event))

                    except asyncio.CancelledError:
                        pass
                    except (ConnectionClosedOK, ConnectionClosedError):
                        # Normal websocket closure or clean disconnect - not an error
                        logger.debug(f"[User] WebSocket closed normally")
                        pass
                    except Exception as e:
                        logger.error(f"[User] Write error: {e}")

                reader_task = asyncio.create_task(read_from_ws())
                writer_task = asyncio.create_task(write_to_ws())

                await reader_task
                writer_task.cancel()
                try:
                    await writer_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.error(f"[User] Connection failed: {e}")
            raise

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
                # Handle evaluation_criteria as strings (not dictionaries)
                criterion_text = criterion if isinstance(criterion, str) else criterion.get("expected", "") if isinstance(criterion, dict) else str(criterion)
                
                criterion_result = {
                    "criterion_id": i + 1,
                    "type": "text_match" if isinstance(criterion, str) else criterion.get("type", "unknown") if isinstance(criterion, dict) else "unknown",
                    "expected": criterion_text,
                    "passed": True,  # Mock pass for now
                    "score": 1.0,
                    "details": f"Mock evaluation of criterion: {criterion_text}"
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
