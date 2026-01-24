-- 05_rpc_functions.sql
-- Secure RPC functions for Move Recommendation App (Step 5 of 5)

-- ============================================================
-- 1. USERNAME AVAILABILITY CHECK
-- ============================================================
-- FIX Problem 2: Client-side username checks are insecure under RLS.
-- This function allows clients to check availability without direct table access.

CREATE OR REPLACE FUNCTION check_username_availability(username_input TEXT)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER -- Runs with elevated privileges to bypass RLS for this specific check
SET search_path = public, pg_temp -- Security best practice
AS $$
BEGIN
  -- Validate input length (defensive programming)
  IF username_input IS NULL OR char_length(username_input) < 3 OR char_length(username_input) > 30 THEN
    RETURN FALSE;
  END IF;

  -- Check existence (case-insensitive)
  -- Returns TRUE if available (count is 0), FALSE if taken
  RETURN NOT EXISTS (
    SELECT 1 
    FROM profiles 
    WHERE LOWER(username) = LOWER(username_input)
  );
END;
$$;

-- Grant execute permission to authenticated users only
REVOKE EXECUTE ON FUNCTION check_username_availability(TEXT) FROM public;
GRANT EXECUTE ON FUNCTION check_username_availability(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION check_username_availability(TEXT) TO service_role;

COMMENT ON FUNCTION check_username_availability IS 'Securely checks if a username is available. Returns TRUE if available.';


-- ============================================================
-- 2. GET USER PROFILE (Helper)
-- ============================================================
-- FIX Problem 1: Safe profile fetching that doesn't deadlock.
-- Useful wrapper for consistent profile retrieval.

CREATE OR REPLACE FUNCTION get_user_profile(user_id_input UUID)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  result JSONB;
BEGIN
  -- Verify the requester is asking for their own profile OR is service role
  IF auth.uid() != user_id_input AND auth.role() != 'service_role' THEN
    RAISE EXCEPTION 'Access Denied';
  END IF;

  SELECT to_jsonb(p.*) INTO result
  FROM profiles p
  WHERE p.id = user_id_input;

  RETURN result;
END;
$$;

REVOKE EXECUTE ON FUNCTION get_user_profile(UUID) FROM public;
GRANT EXECUTE ON FUNCTION get_user_profile(UUID) TO service_role;
GRANT EXECUTE ON FUNCTION get_user_profile(UUID) TO authenticated;
