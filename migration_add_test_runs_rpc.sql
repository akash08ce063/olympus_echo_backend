-- RPC function to fetch test runs with test case results in a single query
-- This replaces multiple sequential queries with one optimized database call

CREATE OR REPLACE FUNCTION get_test_runs_with_results(
    p_user_id UUID,
    p_suite_id UUID DEFAULT NULL,
    p_limit INTEGER DEFAULT 50,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    -- Test run fields
    run_id UUID,
    test_suite_id UUID,
    status TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    total_test_cases INTEGER,
    passed_count INTEGER,
    failed_count INTEGER,
    alert_count INTEGER,
    -- Test case result fields
    result_id UUID,
    test_case_id UUID,
    result_status TEXT,
    concurrent_calls INTEGER,
    recording_file_url TEXT,
    error_message TEXT,
    evaluation_result JSONB,
    conversation_logs JSONB,
    result_started_at TIMESTAMPTZ,
    result_completed_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH limited_runs AS (
        SELECT trh.id
        FROM test_run_history trh
        WHERE 
            trh.user_id = p_user_id
            AND (p_suite_id IS NULL OR trh.test_suite_id = p_suite_id)
        ORDER BY trh.started_at DESC
        LIMIT p_limit
        OFFSET p_offset
    )
    SELECT 
        trh.id AS run_id,
        trh.test_suite_id,
        trh.status::TEXT,
        trh.started_at::TIMESTAMPTZ,
        trh.completed_at::TIMESTAMPTZ,
        trh.total_test_cases,
        trh.passed_count,
        trh.failed_count,
        trh.alert_count,
        tcr.id AS result_id,
        tcr.test_case_id,
        tcr.status::TEXT AS result_status,
        COALESCE(tcr.concurrent_calls, 1) AS concurrent_calls,
        tcr.recording_file_url::TEXT,
        tcr.error_message::TEXT,
        tcr.evaluation_result,
        tcr.conversation_logs,
        tcr.started_at::TIMESTAMPTZ AS result_started_at,
        tcr.completed_at::TIMESTAMPTZ AS result_completed_at
    FROM limited_runs lr
    INNER JOIN test_run_history trh ON trh.id = lr.id
    LEFT JOIN test_case_results tcr ON tcr.test_run_id = trh.id
    ORDER BY trh.started_at DESC, tcr.created_at ASC;
END;
$$;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION get_test_runs_with_results(UUID, UUID, INTEGER, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION get_test_runs_with_results(UUID, UUID, INTEGER, INTEGER) TO anon;
