-- Migration: Add phone_numbers JSONB column to user_agents table
-- Run this SQL in your Supabase SQL editor or database console

-- Add phone_numbers column to store phone numbers used for phone-type tests.
-- The value is a JSON object of the form:
--   { "phone_numbers": ["+15551234567", "+15557654321"] }

ALTER TABLE user_agents
ADD COLUMN IF NOT EXISTS phone_numbers JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN user_agents.phone_numbers IS
'JSON config for phone tests. Expected shape: { "phone_numbers": ["+1555...", ...] }';

