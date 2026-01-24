-- diagnostic_deep_dive.sql
-- Run this in Supabase SQL Editor to pinpoint the username check failure

-- Step 1: Check Permissions (Must return 'authenticated | EXECUTE')
SELECT grantee, privilege_type 
FROM information_schema.role_routine_grants 
WHERE routine_name = 'check_username_availability';

-- Step 2: Check Function Definition (Must define SECURITY DEFINER and query public.profiles)
SELECT pg_get_functiondef('check_username_availability(text)'::regprocedure);

-- Step 3: Test Direct Execution (Should return true/false, NOT error)
SELECT check_username_availability('squadslayer00');
