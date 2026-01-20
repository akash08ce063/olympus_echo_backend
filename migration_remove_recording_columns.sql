-- Migration: Remove recording_file_id and wav_file_ids columns from test_case_results table
-- Run this SQL in your Supabase SQL editor or database console

-- Remove recording_file_id column (we use recording_file_url instead)
ALTER TABLE test_case_results DROP COLUMN IF EXISTS recording_file_id;

-- Remove wav_file_ids column (we can reconstruct from recording_file_url if needed)
ALTER TABLE test_case_results DROP COLUMN IF EXISTS wav_file_ids;

