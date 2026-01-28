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
    List test runs for a user, grouped by test_run_id with all test case results.
    Uses RPC function for fast single-query data fetching.
    Transcripts are fetched from Pranthora in parallel by result_id (request_id) and included per test case result.

    Args:
        user_id: User ID
        suite_id: Optional test suite ID to filter by
        limit: Maximum number of results
        offset: Number of results to skip

    Returns:
        Grouped test runs with test case results, recordings, and transcripts (from Pranthora).
    """
    from data_layer.supabase_client import get_supabase_client

    supabase_client = await get_supabase_client()

    try:
        # Use RPC function to fetch all data in a single optimized query
        rpc_params = {
            "p_user_id": str(user_id),
            "p_limit": limit,
            "p_offset": offset,
        }
        if suite_id:
            rpc_params["p_suite_id"] = str(suite_id)
        else:
            rpc_params["p_suite_id"] = None

        logger.info(f"Calling RPC function get_test_runs_with_results with params: {rpc_params}")
        rpc_results = await supabase_client.call_rpc_function(
            "get_test_runs_with_results", rpc_params
        )

        if not rpc_results:
            return {"total": 0, "runs": []}

        # Group results by run_id
        runs_map = {}
        for row in rpc_results:
            run_id = str(row["run_id"])
            
            # Initialize run if not exists
            if run_id not in runs_map:
                runs_map[run_id] = {
                    "id": run_id,
                    "test_suite_id": str(row["test_suite_id"]) if row["test_suite_id"] else None,
                    "status": row["status"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "total_test_cases": row["total_test_cases"],
                    "passed_count": row["passed_count"],
                    "failed_count": row["failed_count"],
                    "alert_count": row["alert_count"],
                    "test_case_results": [],
                }

            # Add test case result if exists
            if row["result_id"]:
                test_case_result = {
                    "result_id": str(row["result_id"]),
                    "test_case_id": str(row["test_case_id"]) if row["test_case_id"] else None,
                    "concurrent_calls": row["concurrent_calls"] or 1,
                    "call_recordings": [],
                    "recording_url": row["recording_file_url"],
                    "status": row["result_status"],
                    "error_message": row["error_message"],
                    "evaluation_result": row["evaluation_result"],
                    "started_at": row["result_started_at"],
                    "completed_at": row["result_completed_at"],
                }

                # Extract wav_file_ids from conversation_logs for signed URL generation
                conversation_logs = row.get("conversation_logs", []) or []
                wav_file_ids = []
                if isinstance(conversation_logs, list):
                    for log_entry in conversation_logs:
                        if (
                            isinstance(log_entry, dict)
                            and log_entry.get("type") == "recording_metadata"
                            and "wav_file_ids" in log_entry
                        ):
                            wav_file_ids = log_entry.get("wav_file_ids", [])
                            break

                # If we have wav_file_ids, we'll generate signed URLs in batch
                if wav_file_ids and len(wav_file_ids) > 0:
                    test_case_result["wav_file_ids"] = wav_file_ids
                    test_case_result["test_case_id_for_urls"] = str(row["test_case_id"])

                runs_map[run_id]["test_case_results"].append(test_case_result)

        # Collect all result_ids for parallel transcript fetch (result_id is request_id in Pranthora)
        all_result_ids = []
        for run_data in runs_map.values():
            for result in run_data["test_case_results"]:
                if result.get("result_id"):
                    all_result_ids.append(result["result_id"])

        async def fetch_transcript_safe(pranthora_client: PranthoraApiClient, req_id: str):
            """Fetch transcript for one result_id; return (result_id, transcript_or_none). Never raise."""
            try:
                call_logs = await pranthora_client.get_call_logs(req_id)
                transcript = call_logs.get("call_transcript") if call_logs else None
                return (req_id, transcript)
            except Exception as e:
                logger.debug(f"Transcript fetch failed for result_id={req_id}: {e}")
                return (req_id, None)

        # Batch generate signed URLs and fetch transcripts from Pranthora in parallel
        all_url_tasks = []
        url_task_metadata = []  # Track metadata for each URL task in order

        for run_id, run_data in runs_map.items():
            for result in run_data["test_case_results"]:
                if "wav_file_ids" in result:
                    test_case_id = result["test_case_id_for_urls"]
                    wav_file_ids = result["wav_file_ids"]

                    for idx, file_id in enumerate(wav_file_ids):
                        call_num = idx + 1
                        file_name = f"test_case_{test_case_id}_call_{call_num}_recording.wav"
                        file_path = f"{file_id}_{file_name}"

                        # Store metadata for this task
                        url_task_metadata.append({
                            "run_id": run_id,
                            "result_id": result["result_id"],
                            "call_num": call_num,
                            "file_id": file_id,
                        })

                        all_url_tasks.append(
                            supabase_client.create_signed_url("recording_files", file_path, 3600)
                        )

        # Run signed URL generation and Pranthora transcript fetches in parallel
        signed_urls = []
        transcript_by_result_id = {}
        if all_result_ids:
            async with PranthoraApiClient() as pranthora_client:
                transcript_tasks = [
                    fetch_transcript_safe(pranthora_client, rid) for rid in all_result_ids
                ]
                if all_url_tasks:
                    logger.info(
                        f"Generating {len(all_url_tasks)} signed URLs and {len(all_result_ids)} transcripts in parallel..."
                    )
                    url_results, trans_results = await asyncio.gather(
                        asyncio.gather(*all_url_tasks, return_exceptions=True),
                        asyncio.gather(*transcript_tasks, return_exceptions=True),
                    )
                    signed_urls = list(url_results)
                else:
                    logger.info(f"Fetching {len(all_result_ids)} transcripts from Pranthora in parallel...")
                    trans_results = await asyncio.gather(*transcript_tasks, return_exceptions=True)
                for outcome in trans_results:
                    if isinstance(outcome, Exception):
                        continue
                    rid, transcript = outcome
                    transcript_by_result_id[rid] = transcript
        elif all_url_tasks:
            logger.info(f"Generating {len(all_url_tasks)} signed URLs in parallel...")
            signed_urls = await asyncio.gather(*all_url_tasks, return_exceptions=True)

        # Map signed URLs back to results (call_recordings)
        if all_url_tasks and signed_urls:
            result_urls_map = defaultdict(list)
            for idx, signed_url in enumerate(signed_urls):
                if idx < len(url_task_metadata):
                    metadata = url_task_metadata[idx]
                    if signed_url and not isinstance(signed_url, Exception):
                        result_urls_map[metadata["result_id"]].append({
                            "call_number": metadata["call_num"],
                            "recording_url": signed_url,
                            "file_id": metadata["file_id"],
                        })
            for run_id, run_data in runs_map.items():
                for result in run_data["test_case_results"]:
                    if "wav_file_ids" in result:
                        result_id = result["result_id"]
                        call_recordings = result_urls_map.get(result_id, [])
                        call_recordings.sort(key=lambda x: x["call_number"])
                        result["call_recordings"] = call_recordings
                        result["recording_url"] = (
                            call_recordings[0]["recording_url"] if call_recordings else None
                        )
                        result.pop("wav_file_ids", None)
                        result.pop("test_case_id_for_urls", None)

        # Attach transcript to each test case result (from Pranthora by result_id = request_id)
        for run_data in runs_map.values():
            for result in run_data["test_case_results"]:
                result["transcript"] = transcript_by_result_id.get(result["result_id"])

        # Convert runs_map to list
        grouped_runs = list(runs_map.values())

        logger.info(
            f"✅ Fetched {len(grouped_runs)} test runs with results using RPC function in milliseconds"
        )

        return {"total": len(grouped_runs), "runs": grouped_runs}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing test runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list test runs: {str(e)}")


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
            test_run = await run_service.get_test_run(result.get("test_run_id"))
            if not test_run or test_run.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="You don't have permission to view call logs for this test case result",
                )
        else:
            # Test case result doesn't exist, but we'll still try to fetch from Pranthora
            # The request_id might be valid even if test_case_result wasn't created yet
            logger.debug(
                f"Test case result '{request_id}' not found, attempting to fetch from Pranthora directly"
            )
    finally:
        await run_service.close()

    # Fetch call logs from Pranthora API using request_id (which is test_case_result.id - primary key)
    logger.info(
        f"\n\n\nFetching transcript using test_case_result.id (request_id): {request_id}\n\n\n"
    )
    try:
        async with PranthoraApiClient() as pranthora_client:
            # Try fetching with a small retry (similar to evaluation)
            call_logs = None
            for attempt in range(3):  # Try up to 3 times with 1 second delay
                try:
                    call_logs = await pranthora_client.get_call_logs(request_id)
                    if call_logs and call_logs.get("call_transcript"):
                        break
                    elif call_logs:
                        # Call logs exist but no transcript yet - wait and retry
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                except Exception as fetch_error:
                    if attempt < 2:
                        logger.debug(
                            f"Attempt {attempt + 1}/3 failed for {request_id}, retrying..."
                        )
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise fetch_error

            if call_logs and call_logs.get("call_transcript"):
                logger.info(
                    f"✅ Successfully fetched transcript for test_case_result {request_id}: {len(call_logs.get('call_transcript', []))} messages"
                )
                return {"test_case_result_id": request_id, "call_logs": call_logs}
            else:
                logger.warning(f"❌ No call logs found for test_case_result {request_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Call session not found for request_id '{request_id}' - test may not have completed or may have failed",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Error fetching call logs from Pranthora API for test_case_result {request_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve call logs from Pranthora: {str(e)}"
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

        test_run = await run_service.get_test_run(result.get("test_run_id"))
        if not test_run or test_run.user_id != user_id:
            raise HTTPException(
                status_code=403, detail="You don't have permission to access this recording"
            )

        test_case_id = result.get("test_case_id")
        conversation_logs = result.get("conversation_logs", []) or []
        wav_file_ids = []
        if isinstance(conversation_logs, list):
            for log_entry in conversation_logs:
                if (
                    isinstance(log_entry, dict)
                    and log_entry.get("type") == "recording_metadata"
                    and "wav_file_ids" in log_entry
                ):
                    wav_file_ids = log_entry.get("wav_file_ids", [])
                    break
        concurrent_calls = result.get("concurrent_calls", 1) or 1
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
                    signed_url = await supabase_client.create_signed_url(
                        "recording_files", file_path, 3600
                    )
                    if signed_url:
                        recording_urls.append(
                            {
                                "call_number": call_num,
                                "recording_url": signed_url,
                                "file_id": file_id,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to generate URL for call {call_num}: {e}")

            if not recording_urls:
                raise HTTPException(
                    status_code=404, detail="No recording files found for this test result"
                )

            return {
                "result_id": str(result_id),
                "test_case_id": str(test_case_id),
                "concurrent_calls": concurrent_calls,
                "recordings": recording_urls,
                "expires_in": 3600,
            }

        recording_file_url = result.get("recording_file_url")
        if not recording_file_url:
            raise HTTPException(
                status_code=404, detail="No recording file found for this test result"
            )

        return {
            "result_id": str(result_id),
            "test_case_id": str(test_case_id),
            "concurrent_calls": 1,
            "recordings": [{"call_number": 1, "recording_url": recording_file_url}],
            "recording_url": recording_file_url,
            "expires_in": 3600,
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

        results = (
            await client.table("test_case_results")
            .select(
                "id, test_case_id, recording_file_url, status, test_run_id, concurrent_calls, conversation_logs"
            )
            .eq("test_suite_id", str(suite_id))
            .order("created_at", desc=True)
            .execute()
        )

        recordings = []
        for r in results.data or []:
            test_case_id = r["test_case_id"]
            conversation_logs = r.get("conversation_logs", []) or []
            wav_file_ids = []
            if isinstance(conversation_logs, list):
                for log_entry in conversation_logs:
                    if (
                        isinstance(log_entry, dict)
                        and log_entry.get("type") == "recording_metadata"
                        and "wav_file_ids" in log_entry
                    ):
                        wav_file_ids = log_entry.get("wav_file_ids", [])
                        break
            concurrent_calls = r.get("concurrent_calls", 1) or 1

            if wav_file_ids and len(wav_file_ids) > 0:
                call_recordings = []
                for idx, file_id in enumerate(wav_file_ids):
                    call_num = idx + 1
                    file_name = f"test_case_{test_case_id}_call_{call_num}_recording.wav"
                    file_path = f"{file_id}_{file_name}"

                    try:
                        signed_url = await supabase_client.create_signed_url(
                            "recording_files", file_path, 3600
                        )
                        if signed_url:
                            call_recordings.append(
                                {
                                    "call_number": call_num,
                                    "recording_url": signed_url,
                                    "file_id": file_id,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to generate URL for call {call_num}: {e}")

                if call_recordings:
                    recordings.append(
                        {
                            "result_id": r["id"],
                            "test_case_id": test_case_id,
                            "concurrent_calls": concurrent_calls,
                            "call_recordings": call_recordings,
                            "recording_url": call_recordings[0]["recording_url"],
                            "status": r["status"],
                            "run_id": r.get("test_run_id"),
                        }
                    )
            elif r.get("recording_file_url"):
                recordings.append(
                    {
                        "result_id": r["id"],
                        "test_case_id": test_case_id,
                        "concurrent_calls": 1,
                        "call_recordings": [
                            {"call_number": 1, "recording_url": r["recording_file_url"]}
                        ],
                        "recording_url": r["recording_file_url"],
                        "status": r["status"],
                        "run_id": r.get("test_run_id"),
                    }
                )

        return {"suite_id": str(suite_id), "recordings": recordings}

    finally:
        await suite_service.close()
