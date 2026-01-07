"""
Test history service.

This module provides operations for test run history, test case results, and related entities.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
import wave
import io

from services.database_service import DatabaseService
from services.recording_storage_service import RecordingStorageService
from models.test_suite_models import (
    TestRunHistory, TestCaseResult, TestAlert, TestRunWithResults, TestCaseResultWithAlerts,
    TestRunHistoryBase, TestCaseResultBase
)
from telemetrics.logger import logger


class TestRunHistoryService(DatabaseService[TestRunHistory]):
    """Service for test run history read operations."""

    def __init__(self):
        super().__init__("test_run_history")

    async def get_test_run(self, run_id: UUID) -> Optional[TestRunHistory]:
        """Get a test run by ID."""
        result = await self.get_by_id(run_id)
        if result:
            return TestRunHistory(**result)
        return None

    async def get_test_runs_by_suite(
        self, suite_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[TestRunHistory]:
        """Get test runs for a test suite."""
        supabase_client = await self._get_client()

        try:
            results = await supabase_client.select(
                "test_run_history",
                filters={"test_suite_id": str(suite_id)},
                order_by="started_at.desc",
                limit=limit,
                offset=offset
            )

            if not results:
                return []

            return [TestRunHistory(**result) for result in results]
        except Exception as e:
            logger.error(f"Error fetching test runs for suite {suite_id}: {e}")
            raise

    async def get_test_runs_by_user(
        self, user_id: UUID, limit: int = 100, offset: int = 0
    ) -> List[TestRunHistory]:
        """Get test runs for a user."""
        results = await self.get_all_by_user(user_id, limit, offset)
        # Filter out records with null test_suite_id to avoid validation errors
        filtered_results = [result for result in results if result.get('test_suite_id') is not None]
        return [TestRunHistory(**result) for result in filtered_results]

    async def get_test_run_with_results(self, run_id: UUID) -> Optional[TestRunWithResults]:
        """Get a test run with all its results."""
        # Get the test run
        test_run = await self.get_test_run(run_id)
        if not test_run:
            return None

        # Get test case results
        results_service = TestCaseResultService()
        test_case_results = await results_service.get_results_by_run(run_id)

        return TestRunWithResults(
            **test_run.model_dump(),
            test_case_results=test_case_results
        )


class TestCaseResultService(DatabaseService[TestCaseResult]):
    """Service for test case result operations."""

    def __init__(self):
        super().__init__("test_case_results")
        self.recording_service = RecordingStorageService()

    async def get_result(self, result_id: UUID) -> Optional[TestCaseResult]:
        """Get a test case result by ID."""
        result = await self.get_by_id(result_id)
        if result:
            return TestCaseResult(**result)
        return None

    async def get_results_by_run(self, run_id: UUID) -> List[TestCaseResult]:
        """Get all results for a test run."""
        supabase_client = await self._get_client()

        try:
            results = await supabase_client.select(
                "test_case_results",
                filters={"test_run_id": str(run_id)},
                order_by="started_at"
            )

            if not results:
                return []

            return [TestCaseResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Error fetching results for run {run_id}: {e}")
            raise

    async def get_results_by_case(self, case_id: UUID, limit: int = 100) -> List[TestCaseResult]:
        """Get results for a specific test case."""
        supabase_client = await self._get_client()

        try:
            results = await supabase_client.select(
                "test_case_results",
                filters={"test_case_id": str(case_id)},
                order_by="started_at.desc",
                limit=limit
            )

            if not results:
                return []

            return [TestCaseResult(**result) for result in results]
        except Exception as e:
            logger.error(f"Error fetching results for case {case_id}: {e}")
            raise

    async def get_result_with_alerts(self, result_id: UUID) -> Optional[TestCaseResultWithAlerts]:
        """Get a test case result with all its alerts."""
        # Get the result
        test_result = await self.get_result(result_id)
        if not test_result:
            return None

        # Get alerts
        alerts_service = TestAlertService()
        alerts = await alerts_service.get_alerts_by_result(result_id)

        return TestCaseResultWithAlerts(
            **test_result.model_dump(),
            alerts=alerts
        )

    async def create_test_case_result_with_recording(
        self,
        test_run_id: UUID,
        test_case_id: UUID,
        test_suite_id: UUID,
        status: str,
        pcm_frames: bytes,
        sample_rate: int = 16000,
        num_channels: int = 2,  # Default to stereo for combined agent audio
        conversation_logs: Optional[List[Dict[str, Any]]] = None,
        evaluation_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        concurrent_calls: int = 1,
        wav_file_ids: Optional[List[str]] = None
    ) -> UUID:
        """
        Create a test case result and upload recording file(s) to Supabase storage.

        Args:
            test_run_id: ID of the test run
            test_case_id: ID of the test case
            test_suite_id: ID of the test suite
            status: Result status (pass, fail, alert)
            pcm_frames: Raw PCM audio data
            sample_rate: Audio sample rate (default: 16000)
            num_channels: Number of audio channels (1=mono, 2=stereo, default: 2 for combined agent audio)
            conversation_logs: Optional conversation logs
            evaluation_result: Optional evaluation result from user agent
            error_message: Optional error message
            concurrent_calls: Number of concurrent calls (default: 1)
            wav_file_ids: List of pre-uploaded WAV file IDs for concurrent calls

        Returns:
            UUID of the created test case result
        """
        try:
            result_data = {
                "test_run_id": str(test_run_id),
                "test_case_id": str(test_case_id),
                "test_suite_id": str(test_suite_id),
                "status": status,
                "conversation_logs": conversation_logs,
                "evaluation_result": evaluation_result,
                "error_message": error_message
            }

            # Try to add concurrent calls support (will work if columns exist)
            concurrent_supported = False
            try:
                result_data["concurrent_calls"] = concurrent_calls
                concurrent_supported = True
            except Exception:
                logger.warning("Concurrent calls not supported in database, using legacy mode")

            # Handle recording file IDs based on concurrent calls
            if concurrent_supported and wav_file_ids and len(wav_file_ids) > 0:
                # For concurrent calls, use pre-uploaded file IDs
                try:
                    result_data["wav_file_ids"] = wav_file_ids
                    if len(wav_file_ids) == 1:
                        # For backward compatibility, also set single recording_file_id
                        result_data["recording_file_id"] = wav_file_ids[0]
                    logger.info(f"Using pre-uploaded recording files for concurrent calls: {wav_file_ids}")
                except Exception as e:
                    logger.warning(f"Could not store wav_file_ids, falling back to legacy: {e}")
                    concurrent_supported = False

            # Recording files are now uploaded at the test suite level in test_execution_service.py
            # No need to upload individual test case recordings here

            result_id = await self.create(result_data)
            logger.info(f"Created test case result: {result_id} with {concurrent_calls} concurrent call(s)")
            return result_id

        except Exception as e:
            logger.error(f"Error creating test case result with recording: {e}")
            raise

    async def get_result_by_test_run_and_case(self, test_run_id: UUID, test_case_id: UUID) -> Optional[TestCaseResult]:
        """
        Get a test case result by test run ID and test case ID.

        Args:
            test_run_id: ID of the test run
            test_case_id: ID of the test case

        Returns:
            TestCaseResult if found, None otherwise
        """
        try:
            supabase_client = await self._get_client()

            results = await supabase_client.select(
                "test_case_results",
                filters={
                    "test_run_id": str(test_run_id),
                    "test_case_id": str(test_case_id)
                },
                limit=1
            )

            if results and len(results) > 0:
                return TestCaseResult(**results[0])
            return None

        except Exception as e:
            logger.error(f"Error fetching test case result for run {test_run_id} and case {test_case_id}: {e}")
            return None

    async def update_test_case_result_status(
        self,
        result_id: UUID,
        status: str,
        conversation_logs: Optional[List[Dict[str, Any]]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the status and related fields of an existing test case result.

        Args:
            result_id: ID of the test case result to update
            status: New status ('running', 'completed', 'failed')
            conversation_logs: Updated conversation logs (optional)
            error_message: Error message if status is 'failed' (optional)

        Returns:
            True if update was successful, False otherwise
        """
        try:
            update_data = {"status": status}

            if conversation_logs is not None:
                update_data["conversation_logs"] = conversation_logs

            if error_message is not None:
                update_data["error_message"] = error_message

            success = await self.update(result_id, update_data)
            if success:
                logger.info(f"Updated test case result {result_id} status to {status}")
            else:
                logger.error(f"Failed to update test case result {result_id} status to {status}")
            return success

        except Exception as e:
            logger.error(f"Error updating test case result {result_id}: {e}")
            return False


class TestAlertService(DatabaseService[TestAlert]):
    """Service for test alert read operations."""

    def __init__(self):
        super().__init__("test_alerts")

    async def get_alert(self, alert_id: UUID) -> Optional[TestAlert]:
        """Get a test alert by ID."""
        result = await self.get_by_id(alert_id)
        if result:
            return TestAlert(**result)
        return None

    async def get_alerts_by_result(self, result_id: UUID) -> List[TestAlert]:
        """Get all alerts for a test case result."""
        supabase_client = await self._get_client()

        try:
            results = await supabase_client.select(
                "test_alerts",
                filters={"test_case_result_id": str(result_id)},
                order_by="created_at"
            )

            if not results:
                return []

            return [TestAlert(**result) for result in results]
        except Exception as e:
            logger.error(f"Error fetching alerts for result {result_id}: {e}")
            raise

    async def get_recording_file(self, result_id: UUID) -> Optional[bytes]:
        """
        Download the recording file for a test case result.

        Args:
            result_id: ID of the test case result

        Returns:
            Recording file content as bytes, or None if not found
        """
        try:
            # Get the test case result
            result = await self.get_result(result_id)
            if not result or not result.recording_file_id:
                return None

            # Download the recording file
            file_content = await self.recording_service.download_recording_file(
                file_id=result.recording_file_id,
                file_name=f"test_case_{result.test_case_id}.wav"
            )

            return file_content

        except Exception as e:
            logger.error(f"Error downloading recording file for result {result_id}: {e}")
            return None

    async def get_alerts_by_severity(self, severity: str, limit: int = 100) -> List[TestAlert]:
        """Get alerts by severity level."""
        supabase_client = await self._get_client()

        try:
            results = await supabase_client.select(
                "test_alerts",
                filters={"severity": severity},
                order_by="created_at.desc",
                limit=limit
            )

            if not results:
                return []

            return [TestAlert(**result) for result in results]
        except Exception as e:
            logger.error(f"Error fetching alerts by severity {severity}: {e}")
            raise
