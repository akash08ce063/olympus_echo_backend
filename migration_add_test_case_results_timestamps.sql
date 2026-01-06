-- Add missing timestamp columns to test_case_results table
-- This fixes the error: "Could not find the 'created_at' column of 'test_case_results' in the schema cache"

ALTER TABLE test_case_results 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE test_case_results 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- Add missing started_at and completed_at columns if they don't exist
ALTER TABLE test_case_results 
ADD COLUMN IF NOT EXISTS started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

ALTER TABLE test_case_results 
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

-- Update existing records to have timestamps if they're null
UPDATE test_case_results 
SET created_at = NOW() 
WHERE created_at IS NULL;

UPDATE test_case_results 
SET updated_at = NOW() 
WHERE updated_at IS NULL;

UPDATE test_case_results 
SET started_at = created_at 
WHERE started_at IS NULL;

