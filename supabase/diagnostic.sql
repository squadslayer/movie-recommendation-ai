-- diagnostic.sql
-- Check if the function exists
SELECT routine_name, routine_type 
FROM information_schema.routines 
WHERE routine_name = 'check_username_availability';

-- Check permissions
SELECT grantee, privilege_type 
FROM information_schema.role_routine_grants 
WHERE routine_name = 'check_username_availability';

-- Test the function (if it exists)
-- This might fail if the function doesn't exist, which is fine (diagnostic).
DO $$
BEGIN
    PERFORM check_username_availability('test_user');
    RAISE NOTICE 'Function execution successful';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Function execution failed: %', SQLERRM;
END $$;

-- Check if 'profiles' table exists
SELECT to_regclass('public.profiles');
