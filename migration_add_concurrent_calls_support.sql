-- Add concurrent calls support to test_case_results table
-- This fixes the error: "Could not find the 'concurrent_calls' column of 'test_case_results' in the schema cache"

-- Add concurrent_calls column (defaults to 1 for backward compatibility)
ALTER TABLE test_case_results
ADD COLUMN IF NOT EXISTS concurrent_calls INTEGER DEFAULT 1;

-- Add wav_file_ids column for storing multiple recording file IDs (JSON array)
ALTER TABLE test_case_results
ADD COLUMN IF NOT EXISTS wav_file_ids JSONB;

-- Update existing records to have concurrent_calls = 1 if null
UPDATE test_case_results
SET concurrent_calls = 1
WHERE concurrent_calls IS NULL;
