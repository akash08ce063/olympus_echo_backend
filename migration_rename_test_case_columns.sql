-- Migration: Rename columns in test_cases table and delete expected_outcomes column
-- Run this SQL in your Supabase SQL editor or database console

-- Rename 'steps' column to 'goals'
ALTER TABLE test_cases RENAME COLUMN steps TO goals;

-- Rename 'conditions' column to 'evaluation_criteria'
ALTER TABLE test_cases RENAME COLUMN conditions TO evaluation_criteria;

-- Drop the expected_outcomes column
ALTER TABLE test_cases DROP COLUMN IF EXISTS expected_outcomes;

-- Add comments for the new columns
COMMENT ON COLUMN test_cases.goals IS 'Array of test goals/prompts for the test case';
COMMENT ON COLUMN test_cases.evaluation_criteria IS 'Array of evaluation criteria for validating test results';
