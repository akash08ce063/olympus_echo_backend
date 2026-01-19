"""
Test Runs API routes.

This module provides REST API endpoints for viewing test runs, recordings, and transcripts.
Separated from test_execution_routes.py for modular code organization.
"""

import asyncio
from collections import defaultdict
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from services.pranthora_api_client import PranthoraApiClient
from services.test_history_service import TestRunHistoryService, TestCaseResultService
from static_memory_cache import StaticMemoryCache
from telemetrics.logger import logger

router = APIRouter(prefix="/test-runs", tags=["Test Runs"])


@router.get("/")
async def list_test_runs(
    user_id: UUID = Query(..., description="User ID"),
    suite_id: Optional[UUID] = Query(None, description="Filter by test suite ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
):
    """
    List test runs for a user, grouped by test_run_id with all test case results and concurrent calls.

    Args:
        user_id: User ID
        suite_id: Optional test suite ID to filter by
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Grouped test runs with test case results, recordings, and transcripts
    """
    from supabase.client import acreate_client
    from data_layer.supabase_client import get_supabase_client

    run_service = TestRunHistoryService()
    
    try:
        if suite_id:
            runs = await run_service.get_test_runs_by_suite(suite_id, limit, offset)
            runs = [run for run in runs if run.user_id == user_id]
        else:
            runs = await run_service.get_test_runs_by_user(user_id, limit, offset)

        # Get database client for direct queries
        db_config = StaticMemoryCache.get_database_config()
        client = await acreate_client(db_config["supabase_url"], db_config["supabase_key"])
        supabase_client = await get_supabase_client()
        
        # Fetch all test case results for all runs in parallel
        async def fetch_test_case_results(run_id: str):
            """Fetch test case results for a single run."""
            try:
                results = await client.table('test_case_results').select(
                    'id, test_case_id, test_suite_id, status, test_run_id, concurrent_calls, recording_file_url, error_message, evaluation_result, conversation_logs, started_at, completed_at'
                ).eq('test_run_id', run_id).order('created_at', desc=False).execute()
                return run_id, results.data or []
            except Exception as e:
                logger.warning(f"Failed to fetch test case results for run {run_id}: {e}")
                return run_id, []
        
        test_case_tasks = [fetch_test_case_results(str(run.id)) for run in runs]
        test_case_results_map = dict(await asyncio.gather(*test_case_tasks))
        
        # Helper function to generate signed URLs in parallel
        async def generate_signed_url(file_path: str, call_num: int, file_id: str):
            """Generate a signed URL for a recording file."""
            try:
                signed_url = await supabase_client.create_signed_url("recording_files", file_path, 3600)
                if signed_url:
                    return {
                        "call_number": call_num,
                        "recording_url": signed_url,
                        "file_id": file_id
                    }
                return None
            except Exception as e:
                logger.warning(f"Failed to generate URL for call {call_num}: {e}")
                return None
        
        # Group runs with their test case results
        grouped_runs = []
        
        for run in runs:
            run_id = str(run.id)
            results = test_case_results_map.get(run_id, [])
            
            # Group results by test_case_id to handle multiple entries per test case (concurrent calls)
            results_by_test_case = defaultdict(list)
            for r in results:
                test_case_id = r['test_case_id']
                results_by_test_case[test_case_id].append(r)
            
            test_case_results = []
            
            for test_case_id, test_case_result_entries in results_by_test_case.items():
                # Use the first entry for metadata (they should all have the same test_case_id, concurrent_calls, etc.)
                first_entry = test_case_result_entries[0]
                concurrent_calls = first_entry.get('concurrent_calls', len(test_case_result_entries)) or len(test_case_result_entries)
                
                # Extract wav_file_ids from conversation_logs metadata (from first entry)
                conversation_logs = first_entry.get('conversation_logs', []) or []
                wav_file_ids = []
                if isinstance(conversation_logs, list):
                    for log_entry in conversation_logs:
                        if isinstance(log_entry, dict) and log_entry.get('type') == 'recording_metadata' and 'wav_file_ids' in log_entry:
                            wav_file_ids = log_entry.get('wav_file_ids', [])
                            break
                
                # Build call_recordings array for all concurrent calls
                call_recordings = []
                call_transcripts = []
                
                # Fetch transcripts for ALL test_case_result entries (one per concurrent call)
                async def fetch_transcript_for_result(result_entry):
                    """Fetch transcript for a single test_case_result entry."""
                    result_id = result_entry['id']
                    logger.info(f"Fetching transcript using test_case_result.id: {result_id}")
                    try:
                        async with PranthoraApiClient() as pranthora_client:
                            call_logs = await pranthora_client.get_call_logs(result_id)
                            if call_logs and call_logs.get('call_transcript'):
                                logger.info(f"✅ Successfully fetched transcript for test_case_result {result_id}: {len(call_logs.get('call_transcript', []))} messages")
                                return {
                                    "request_id": call_logs.get('request_id'),
                                    "id": call_logs.get('id'),
                                    "call_transcript": call_logs.get('call_transcript', []),
                                    "duration_seconds": call_logs.get('call_duration'),
                                    "created_at": call_logs.get('created_at')
                                }
                            else:
                                logger.warning(f"No transcript found for test_case_result {result_id}")
                                return None
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch transcript for test_case_result {result_id}: {e}")
                        return None
                
                # Fetch all transcripts in parallel
                transcript_tasks = [fetch_transcript_for_result(entry) for entry in test_case_result_entries]
                transcript_results = await asyncio.gather(*transcript_tasks)
                
                # Add successful transcript fetches
                for transcript in transcript_results:
                    if transcript:
                        call_transcripts.append(transcript)
                
                # Build call_recordings from wav_file_ids or individual recording_file_urls
                if wav_file_ids and len(wav_file_ids) > 0:
                    # Generate all signed URLs in parallel
                    url_tasks = []
                    for idx, file_id in enumerate(wav_file_ids):
                        call_num = idx + 1
                        file_name = f"test_case_{test_case_id}_call_{call_num}_recording.wav"
                        file_path = f"{file_id}_{file_name}"
                        url_tasks.append(generate_signed_url(file_path, call_num, file_id))
                    
                    signed_url_results = await asyncio.gather(*url_tasks)
                    
                    # Process results and build call_recordings
                    for idx, signed_url_result in enumerate(signed_url_results):
                        if signed_url_result:
                            call_recordings.append(signed_url_result)
                else:
                    # Fallback: use recording_file_url from each entry
                    for idx, entry in enumerate(test_case_result_entries):
                        if entry.get('recording_file_url'):
                            call_recordings.append({
                                "call_number": idx + 1,
                                "recording_url": entry['recording_file_url']
                            })
                
                # Use first entry's result_id for backward compatibility
                primary_result_id = first_entry['id']
                
                test_case_results.append({
                    "result_id": primary_result_id,
                    "test_case_id": test_case_id,
                    "concurrent_calls": concurrent_calls,
                    "call_recordings": call_recordings,
                    "call_transcripts": call_transcripts,  # Now contains transcripts for ALL concurrent calls
                    "recording_url": call_recordings[0]["recording_url"] if call_recordings else None,
                    "status": first_entry['status'],
                    "error_message": first_entry.get('error_message'),
                    "evaluation_result": first_entry.get('evaluation_result'),
                    "started_at": first_entry.get('started_at'),
                    "completed_at": first_entry.get('completed_at')
                })
            
            grouped_runs.append({
                "id": run_id,
                "test_suite_id": str(run.test_suite_id),
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "total_test_cases": run.total_test_cases,
                "passed_count": run.passed_count,
                "failed_count": run.failed_count,
                "alert_count": run.alert_count,
                "test_case_results": test_case_results
            })

        return {
            "total": len(grouped_runs),
            "runs": grouped_runs
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing test runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list test runs: {str(e)}")
    finally:
        await run_service.close()


@router.get("/call-logs/{request_id}")
async def get_call_logs_by_request_id(
    request_id: str,
    user_id: UUID = Query(..., description="User ID for authorization"),
):
    """
    Get call logs/session transcripts for a test case result by its ID (which is the request_id).
    Uses Pranthora API to fetch call session data.

    Args:
        request_id: The test_case_result.id (which is the request_id used for the user agent call)
        user_id: User ID for authorization

    Returns:
        Call session data including transcripts from Pranthora backend
    """
    # Try to verify access through test_case_result if it exists
    result_service = TestCaseResultService()
    run_service = TestRunHistoryService()
    try:
        result = await result_service.get_by_id(UUID(request_id))
        if result:
            # Verify user has access through the test run
            test_run = await run_service.get_test_run(result.get('test_run_id'))
            if not test_run or test_run.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to view call logs for this test case result"
                )
        else:
            # Test case result doesn't exist, but we'll still try to fetch from Pranthora
            # The request_id might be valid even if test_case_result wasn't created yet
            logger.debug(f"Test case result '{request_id}' not found, attempting to fetch from Pranthora directly")
    finally:
        await run_service.close()

    # Fetch call logs from Pranthora API using request_id (which is test_case_result.id - primary key)
    logger.info(f"\n\n\nFetching transcript using test_case_result.id (request_id): {request_id}\n\n\n")
    try:
        async with PranthoraApiClient() as pranthora_client:
            # Try fetching with a small retry (similar to evaluation)
            call_logs = None
            for attempt in range(3):  # Try up to 3 times with 1 second delay
                try:
                    call_logs = await pranthora_client.get_call_logs(request_id)
                    if call_logs and call_logs.get('call_transcript'):
                        break
                    elif call_logs:
                        # Call logs exist but no transcript yet - wait and retry
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                except Exception as fetch_error:
                    if attempt < 2:
                        logger.debug(f"Attempt {attempt + 1}/3 failed for {request_id}, retrying...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise fetch_error
            
            if call_logs and call_logs.get('call_transcript'):
                logger.info(f"✅ Successfully fetched transcript for test_case_result {request_id}: {len(call_logs.get('call_transcript', []))} messages")
                return {
                    "test_case_result_id": request_id,
                    "call_logs": call_logs
                }
            else:
                logger.warning(f"❌ No call logs found for test_case_result {request_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Call session not found for request_id '{request_id}' - test may not have completed or may have failed"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching call logs from Pranthora API for test_case_result {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve call logs from Pranthora: {str(e)}"
        )


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
        call_number: Optional call number for concurrent calls

    Returns:
        Signed URL(s) to download/play the recording(s)
    """
    from data_layer.supabase_client import get_supabase_client

    result_service = TestCaseResultService()
    run_service = TestRunHistoryService()

    try:
        result = await result_service.get_by_id(result_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Test case result '{result_id}' not found")

        test_run = await run_service.get_test_run(result.get('test_run_id'))
        if not test_run or test_run.user_id != user_id:
            raise HTTPException(status_code=403, detail="You don't have permission to access this recording")

        test_case_id = result.get('test_case_id')
        conversation_logs = result.get('conversation_logs', []) or []
        wav_file_ids = []
        if isinstance(conversation_logs, list):
            for log_entry in conversation_logs:
                if isinstance(log_entry, dict) and log_entry.get('type') == 'recording_metadata' and 'wav_file_ids' in log_entry:
                    wav_file_ids = log_entry.get('wav_file_ids', [])
                    break
        concurrent_calls = result.get('concurrent_calls', 1) or 1
        supabase_client = await get_supabase_client()

        if wav_file_ids and len(wav_file_ids) > 0:
            recording_urls = []

            for idx, file_id in enumerate(wav_file_ids):
                call_num = idx + 1
                if call_number is not None and call_num != call_number:
                    continue

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


@router.get("/recordings/suite/{suite_id}")
async def get_recordings_by_suite(
    suite_id: UUID,
    user_id: UUID = Query(..., description="User ID for authorization"),
):
    """Get all recording URLs for a test suite, including all concurrent call recordings."""
    from services.test_suite_service import TestSuiteService
    from supabase.client import acreate_client
    from data_layer.supabase_client import get_supabase_client

    suite_service = TestSuiteService()

    try:
        suite = await suite_service.get_test_suite(suite_id)
        if not suite:
            raise HTTPException(status_code=404, detail="Suite not found")
     
        db_config = StaticMemoryCache.get_database_config()
        client = await acreate_client(db_config["supabase_url"], db_config["supabase_key"])
        supabase_client = await get_supabase_client()
        
        results = await client.table('test_case_results').select(
            'id, test_case_id, recording_file_url, status, test_run_id, concurrent_calls, conversation_logs'
        ).eq('test_suite_id', str(suite_id)).order('created_at', desc=True).execute()

        recordings = []
        for r in results.data or []:
            test_case_id = r['test_case_id']
            conversation_logs = r.get('conversation_logs', []) or []
            wav_file_ids = []
            if isinstance(conversation_logs, list):
                for log_entry in conversation_logs:
                    if isinstance(log_entry, dict) and log_entry.get('type') == 'recording_metadata' and 'wav_file_ids' in log_entry:
                        wav_file_ids = log_entry.get('wav_file_ids', [])
                        break
            concurrent_calls = r.get('concurrent_calls', 1) or 1

            if wav_file_ids and len(wav_file_ids) > 0:
                call_recordings = []
                for idx, file_id in enumerate(wav_file_ids):
                    call_num = idx + 1
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
                        "recording_url": call_recordings[0]["recording_url"],
                        "status": r['status'],
                        "run_id": r.get('test_run_id')
                    })
            elif r.get('recording_file_url'):
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

