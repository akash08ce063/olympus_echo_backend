"""
Test execution API routes.

This module provides REST API endpoints for running test cases and test suites.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Request
from pydantic import BaseModel

from services.test_execution_service import TestExecutionService
from services.test_suite_service import TestSuiteService
from services.test_case_service import TestCaseService
from services.pranthora_api_client import PranthoraApiClient
from telemetrics.logger import logger

router = APIRouter(prefix="/test-execution", tags=["Test Execution"])


# Dependency to get test execution service
async def get_test_execution_service() -> TestExecutionService:
    """Dependency to get test execution service instance."""
    service = TestExecutionService()
    try:
        yield service
    finally:
        await service.close()


class RunTestSuiteRequest(BaseModel):
    """Request model for running a test suite."""
    concurrent_calls: Optional[int] = Query(1, ge=1, le=10, description="Number of concurrent calls (overrides default)")


class RunTestCaseRequest(BaseModel):
    """Request model for running a test case."""
    concurrent_calls: Optional[int] = Query(1, ge=1, le=10, description="Number of concurrent calls (overrides default)")


class TestExecutionResponse(BaseModel):
    """Response model for test execution."""
    success: bool
    test_run_id: UUID
    message: str
    status: dict


@router.post("/run-suite/{suite_id}", response_model=TestExecutionResponse)
async def run_test_suite(
    suite_id: UUID,
    request: RunTestSuiteRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Query(..., description="User ID who is running the test"),
    request_obj: Request = None,  # Add request object to capture headers
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of all active test cases in a test suite.

    Args:
        suite_id: ID of the test suite to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
        request_obj: FastAPI request object to capture headers
        service: Test execution service

    Returns:
        Test execution response with run ID
    """
    test_suite_service = None
    test_case_service = None
    active_cases = []

    try:
        # Validate that the test suite exists and user has access
        test_suite_service = TestSuiteService()
        test_suite = await test_suite_service.get_test_suite(suite_id)
        if not test_suite:
            raise HTTPException(status_code=404, detail=f"Test suite '{suite_id}' not found")

        if test_suite.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to run tests for this test suite"
            )

        # Check if there are active test cases
        test_case_service = TestCaseService()
        active_cases = await test_case_service.get_test_cases_by_suite(suite_id, include_inactive=False)
        if not active_cases:
            raise HTTPException(
                status_code=400,
                detail="No active test cases found in this test suite"
            )

        # Get the request ID from header
        request_id = request_obj.headers.get("x-pranthora-callid")
        if not request_id:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header is required"
            )

        # Start the test execution
        test_run_id = await service.run_test_suite(
            test_suite_id=suite_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls,
            request_id=request_id
        )

        return TestExecutionResponse(
            success=True,
            test_run_id=test_run_id,
            message=f"Started execution of test suite '{suite_id}' with {len(active_cases)} test cases",
            status={
                "test_run_id": str(test_run_id),
                "suite_id": str(suite_id),
                "status": "running",
                "total_test_cases": len(active_cases),
                "concurrent_calls": request.concurrent_calls
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test suite execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test execution: {str(e)}")
    finally:
        if test_suite_service:
            await test_suite_service.close()
        if test_case_service:
            await test_case_service.close()


@router.post("/run-case/{case_id}", response_model=TestExecutionResponse)
async def run_test_case(
    case_id: UUID,
    request: RunTestCaseRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Query(..., description="User ID who is running the test"),
    request_obj: Request = None,  # Add request object to capture headers
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Start execution of a single test case.

    Args:
        case_id: ID of the test case to run
        request: Request parameters
        user_id: User ID (should come from authentication middleware)
        request_obj: FastAPI request object to capture headers
        service: Test execution service

    Returns:
        Test execution response with run ID
    """
    try:
        # Validate that the test case exists and user has access
        test_case_service = TestCaseService()
        try:
            test_case = await test_case_service.get_test_case(case_id)
            if not test_case:
                raise HTTPException(status_code=404, detail=f"Test case '{case_id}' not found")

            # Check user access via test suite
            test_suite_service = TestSuiteService()
            try:
                test_suite = await test_suite_service.get_test_suite(test_case.test_suite_id)
                if not test_suite or test_suite.user_id != user_id:
                    raise HTTPException(
                        status_code=403,
                        detail="You don't have permission to run this test case"
                    )
            finally:
                await test_suite_service.close()

            # Check if test case is active
            if not test_case.is_active:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot run inactive test case"
                )

        finally:
            await test_case_service.close()

        # Get the request ID from header
        request_id = request_obj.headers.get("x-pranthora-callid")
        if not request_id:
            raise HTTPException(
                status_code=400,
                detail="x-pranthora-callid header is required"
            )

        # Start the test execution
        test_run_id = await service.run_single_test_case(
            test_case_id=case_id,
            user_id=user_id,
            concurrent_calls=request.concurrent_calls,
            request_id=request_id
        )

        return TestExecutionResponse(
            success=True,
            test_run_id=test_run_id,
            message=f"Started execution of test case '{case_id}'",
            status={
                "test_run_id": str(test_run_id),
                "case_id": str(case_id),
                "status": "running",
                "concurrent_calls": request.concurrent_calls
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting test case execution: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start test execution: {str(e)}")


@router.get("/status/{run_id}")
async def get_test_run_status(
    run_id: UUID,
    user_id: UUID = Query(..., description="User ID for authorization"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Get the status of a test run.

    Args:
        run_id: Test run ID
        user_id: User ID for authorization
        service: Test execution service

    Returns:
        Test run status information
    """
    try:
        from services.test_history_service import TestRunHistoryService

        run_service = TestRunHistoryService()
        try:
            test_run = await run_service.get_test_run(run_id)
            if not test_run:
                raise HTTPException(status_code=404, detail=f"Test run '{run_id}' not found")

            # Check user access
            if test_run.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to view this test run"
                )

            # Get test run with results
            test_run_with_results = await run_service.get_test_run_with_results(run_id)

            return {
                "test_run_id": str(run_id),
                "status": test_run.status,
                "started_at": test_run.started_at,
                "completed_at": test_run.completed_at,
                "total_test_cases": test_run.total_test_cases,
                "passed_count": test_run.passed_count,
                "failed_count": test_run.failed_count,
                "alert_count": test_run.alert_count,
                "test_case_results": [
                    {
                        "id": str(result.id),
                        "test_case_id": str(result.test_case_id),
                        "status": result.status,
                        "started_at": result.started_at,
                        "completed_at": result.completed_at,
                        "recording_file_id": str(result.recording_file_id) if result.recording_file_id else None,
                        "error_message": result.error_message
                    }
                    for result in (test_run_with_results.test_case_results if test_run_with_results else [])
                ]
            }

        finally:
            await run_service.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting test run status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get test run status: {str(e)}")


@router.get("/runs")
async def list_test_runs(
    user_id: UUID = Query(..., description="User ID"),
    suite_id: Optional[UUID] = Query(None, description="Filter by test suite ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    List test runs for a user.

    Args:
        user_id: User ID
        suite_id: Optional test suite ID to filter by
        limit: Maximum number of results
        offset: Number of results to skip
        service: Test execution service

    Returns:
        List of test runs
    """
    try:
        from services.test_history_service import TestRunHistoryService

        run_service = TestRunHistoryService()
        try:
            if suite_id:
                # Filter by suite
                runs = await run_service.get_test_runs_by_suite(suite_id, limit, offset)
                # Filter by user access
                runs = [run for run in runs if run.user_id == user_id]
            else:
                # Get all runs for user
                runs = await run_service.get_test_runs_by_user(user_id, limit, offset)

            return {
                "total": len(runs),
                "runs": [
                    {
                        "id": str(run.id),
                        "test_suite_id": str(run.test_suite_id),
                        "status": run.status,
                        "started_at": run.started_at,
                        "completed_at": run.completed_at,
                        "total_test_cases": run.total_test_cases,
                        "passed_count": run.passed_count,
                        "failed_count": run.failed_count,
                        "alert_count": run.alert_count
                    }
                    for run in runs
                ]
            }

        finally:
            await run_service.close()

    except Exception as e:
        logger.error(f"Error listing test runs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list test runs: {str(e)}")


@router.get("/call-logs/{request_id}")
async def get_call_logs_by_request_id(
    request_id: str,
    user_id: UUID = Query(..., description="User ID for authorization"),
    service: TestExecutionService = Depends(get_test_execution_service),
):
    """
    Get call logs/session transcripts for a test run by its request ID.

    Args:
        request_id: The request ID (same as test_run_history.id)
        user_id: User ID for authorization
        service: Test execution service

    Returns:
        Call session data including transcripts from Pranthora backend
    """
    try:
        # First verify that the test run exists and user has access
        from services.test_history_service import TestRunHistoryService

        run_service = TestRunHistoryService()
        try:
            test_run = await run_service.get_test_run(request_id)
            if not test_run:
                raise HTTPException(status_code=404, detail=f"Test run '{request_id}' not found")

            # Check user access
            if test_run.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to view call logs for this test run"
                )

        finally:
            await run_service.close()

        # Get call logs from Pranthora backend
        async with PranthoraApiClient() as pranthora_client:
            try:
                # Call the Pranthora call-analytics endpoint to get call logs
                response = await pranthora_client.client.get(
                    f"{pranthora_client.base_url}/api/v1/call-analytics/call-logs/{request_id}",
                    headers={
                        "x-api-key": pranthora_client.api_key,
                        "Content-Type": "application/json"
                    }
                )

                if response.status_code == 200:
                    call_logs_data = response.json()
                    return {
                        "test_run_id": request_id,
                        "call_logs": call_logs_data
                    }
                elif response.status_code == 404:
                    return {
                        "test_run_id": request_id,
                        "call_logs": None,
                        "message": "Call session not found - test may not have completed or may have failed"
                    }
                else:
                    logger.error(f"Failed to get call logs from Pranthora: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to retrieve call logs from Pranthora backend"
                    )

            except Exception as e:
                logger.error(f"Error communicating with Pranthora backend: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to communicate with Pranthora backend: {str(e)}"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get call logs: {str(e)}")


@router.get("/recording/{result_id}")
async def get_recording_url(
    result_id: UUID,
    user_id: UUID = Query(..., description="User ID for authorization"),
    call_number: int = Query(None, description="Call number for concurrent calls (1-indexed)"),
):
    """
    Get signed URL(s) for the recording file(s) of a test case result.

    Args:
        result_id: Test case result ID
        user_id: User ID for authorization
        call_number: Optional call number for concurrent calls (1-indexed). If not provided, returns all recordings.

    Returns:
        Signed URL(s) to download/play the recording(s)
    """
    try:
        from services.test_history_service import TestCaseResultService, TestRunHistoryService
        from services.recording_storage_service import RecordingStorageService
        from data_layer.supabase_client import get_supabase_client

        result_service = TestCaseResultService()
        run_service = TestRunHistoryService()
        recording_service = RecordingStorageService()

        try:
            # Get the test case result
            result = await result_service.get_by_id(result_id)
            if not result:
                raise HTTPException(status_code=404, detail=f"Test case result '{result_id}' not found")

            # Verify user access via test run
            test_run = await run_service.get_test_run(result.get('test_run_id'))
            if not test_run or test_run.user_id != user_id:
                raise HTTPException(status_code=403, detail="You don't have permission to access this recording")

            test_case_id = result.get('test_case_id')
            wav_file_ids = result.get('wav_file_ids', [])
            concurrent_calls = result.get('concurrent_calls', 1)
            supabase_client = await get_supabase_client()

            # If we have wav_file_ids, generate URLs for all/specific calls
            if wav_file_ids and len(wav_file_ids) > 0:
                recording_urls = []

                for idx, file_id in enumerate(wav_file_ids):
                    call_num = idx + 1
                    # Skip if specific call_number requested and this isn't it
                    if call_number is not None and call_num != call_number:
                        continue

                    # Fixed format: test_case_{test_case_id}_call_{call_number}_recording.wav
                    file_name = f"test_case_{test_case_id}_call_{call_num}_recording.wav"
                    file_path = f"{file_id}_{file_name}"

                    try:
                        signed_url = await supabase_client.create_signed_url("recording_files", file_path, 3600)
                        if signed_url:
                            recording_urls.append({
                                "call_number": call_num,
                                "recording_url": signed_url,
                                "file_id": file_id
                            })
                    except Exception as e:
                        logger.warning(f"Failed to generate URL for call {call_num}: {e}")

                if not recording_urls:
                    raise HTTPException(status_code=404, detail="No recording files found for this test result")

                return {
                    "result_id": str(result_id),
                    "test_case_id": str(test_case_id),
                    "concurrent_calls": concurrent_calls,
                    "recordings": recording_urls,
                    "expires_in": 3600
                }

            # Fallback to legacy single recording_file_url
            recording_file_url = result.get('recording_file_url')
            if not recording_file_url:
                raise HTTPException(status_code=404, detail="No recording file found for this test result")

            return {
                "result_id": str(result_id),
                "test_case_id": str(test_case_id),
                "concurrent_calls": 1,
                "recordings": [{"call_number": 1, "recording_url": recording_file_url}],
                "recording_url": recording_file_url,
                "expires_in": 3600
            }

        finally:
            await result_service.close()
            await run_service.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recording URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recording URL: {str(e)}")


@router.get("/recordings/suite/{suite_id}")
async def get_recordings_by_suite(
    suite_id: UUID,
    user_id: UUID = Query(..., description="User ID for authorization"),
):
    """Get all recording URLs for a test suite, including all concurrent call recordings."""
    try:
        from services.test_suite_service import TestSuiteService
        from services.recording_storage_service import RecordingStorageService
        from supabase.client import acreate_client
        from static_memory_cache import StaticMemoryCache
        from data_layer.supabase_client import get_supabase_client

        suite_service = TestSuiteService()
        recording_service = RecordingStorageService()

        try:
            # Verify suite exists and user owns it
            suite = await suite_service.get_test_suite(suite_id)
            if not suite:
                raise HTTPException(status_code=404, detail="Suite not found")
         
            # Get all results for this suite directly
            db_config = StaticMemoryCache.get_database_config()
            client = await acreate_client(db_config["supabase_url"], db_config["supabase_key"])
            supabase_client = await get_supabase_client()
            
            results = await client.table('test_case_results').select(
                'id, test_case_id, recording_file_url, status, test_run_id, wav_file_ids, concurrent_calls'
            ).eq('test_suite_id', str(suite_id)).order('created_at', desc=True).execute()

            recordings = []
            for r in results.data or []:
                test_case_id = r['test_case_id']
                wav_file_ids = r.get('wav_file_ids', [])
                concurrent_calls = r.get('concurrent_calls', 1)

                # If we have wav_file_ids, generate URLs for all calls
                if wav_file_ids and len(wav_file_ids) > 0:
                    call_recordings = []
                    for idx, file_id in enumerate(wav_file_ids):
                        call_num = idx + 1
                        # Fixed format: test_case_{test_case_id}_call_{call_number}_recording.wav
                        file_name = f"test_case_{test_case_id}_call_{call_num}_recording.wav"
                        file_path = f"{file_id}_{file_name}"

                        try:
                            signed_url = await supabase_client.create_signed_url("recording_files", file_path, 3600)
                            if signed_url:
                                call_recordings.append({
                                    "call_number": call_num,
                                    "recording_url": signed_url,
                                    "file_id": file_id
                                })
                        except Exception as e:
                            logger.warning(f"Failed to generate URL for call {call_num}: {e}")

                    if call_recordings:
                        recordings.append({
                            "result_id": r['id'],
                            "test_case_id": test_case_id,
                            "concurrent_calls": concurrent_calls,
                            "call_recordings": call_recordings,
                            "recording_url": call_recordings[0]["recording_url"],  # Backward compat
                            "status": r['status'],
                            "run_id": r.get('test_run_id')
                        })
                elif r.get('recording_file_url'):
                    # Fallback to legacy single recording_file_url
                    recordings.append({
                        "result_id": r['id'],
                        "test_case_id": test_case_id,
                        "concurrent_calls": 1,
                        "call_recordings": [{"call_number": 1, "recording_url": r['recording_file_url']}],
                        "recording_url": r['recording_file_url'],
                        "status": r['status'],
                        "run_id": r.get('test_run_id')
                    })

            return {"suite_id": str(suite_id), "recordings": recordings}

        finally:
            await suite_service.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
