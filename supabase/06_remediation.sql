-- 06_remediation.sql
-- Fixes for Supabase Linter warnings (Step 6 of 6) - IDEMPOTENT & STRICT

-- ============================================================
-- 1. FIX: auth_rls_initplan (Wrap auth.uid() checks)
-- ============================================================

-- PROFILES
-- Drop legacy/previous policies to ensure clean slate
DROP POLICY IF EXISTS "Public profiles are viewable by everyone" ON public.profiles;
DROP POLICY IF EXISTS "Users can insert their own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles; -- Ensure new policy name is also dropped

-- FIX ISSUE 1: Strict Ownership for SELECT (No public reads)
CREATE POLICY "Users can view own profile" 
  ON public.profiles FOR SELECT USING ((select auth.uid()) = id);
  
CREATE POLICY "Users can insert their own profile" 
  ON public.profiles FOR INSERT WITH CHECK ((select auth.uid()) = id);

-- FIX ISSUE 3: UPDATE with WITH CHECK
CREATE POLICY "Users can update own profile" 
  ON public.profiles FOR UPDATE 
  USING ((select auth.uid()) = id)
  WITH CHECK ((select auth.uid()) = id);


-- PREFERENCES
DROP POLICY IF EXISTS "Users can view own preferences" ON public.preferences;
DROP POLICY IF EXISTS "Users can insert own preferences" ON public.preferences;
DROP POLICY IF EXISTS "Users can update own preferences" ON public.preferences;

CREATE POLICY "Users can view own preferences" 
  ON public.preferences FOR SELECT USING ((select auth.uid()) = user_id);
  
CREATE POLICY "Users can insert own preferences" 
  ON public.preferences FOR INSERT WITH CHECK ((select auth.uid()) = user_id);

-- FIX ISSUE 3: UPDATE with WITH CHECK
CREATE POLICY "Users can update own preferences" 
  ON public.preferences FOR UPDATE 
  USING ((select auth.uid()) = user_id)
  WITH CHECK ((select auth.uid()) = user_id);


-- WATCHLIST
DROP POLICY IF EXISTS "Users can view own watchlist" ON public.watchlist;
DROP POLICY IF EXISTS "Users can insert own watchlist" ON public.watchlist;
DROP POLICY IF EXISTS "Users can update own watchlist" ON public.watchlist;
DROP POLICY IF EXISTS "Users can delete own watchlist" ON public.watchlist;
-- Drop potentially aliased policies if they existed in dev cycles
DROP POLICY IF EXISTS "Users can delete own watchlist item" ON public.watchlist; 

CREATE POLICY "Users can view own watchlist" 
  ON public.watchlist FOR SELECT USING ((select auth.uid()) = user_id);
  
CREATE POLICY "Users can insert own watchlist" 
  ON public.watchlist FOR INSERT WITH CHECK ((select auth.uid()) = user_id);

-- FIX ISSUE 3: UPDATE with WITH CHECK
CREATE POLICY "Users can update own watchlist" 
  ON public.watchlist FOR UPDATE 
  USING ((select auth.uid()) = user_id)
  WITH CHECK ((select auth.uid()) = user_id);
  
CREATE POLICY "Users can delete own watchlist" 
  ON public.watchlist FOR DELETE USING ((select auth.uid()) = user_id);


-- ACTIVITY LOG
DROP POLICY IF EXISTS "Users can view own activity" ON public.activity_log;
DROP POLICY IF EXISTS "Users can insert own activity" ON public.activity_log;
DROP POLICY IF EXISTS "Allow anonymous activity logging" ON public.activity_log;

CREATE POLICY "Users can view own activity" 
  ON public.activity_log FOR SELECT USING ((select auth.uid()) = user_id);
  
-- FIX ISSUE 2: No Anonymous Insert (Strict)
CREATE POLICY "Users can insert own activity" 
  ON public.activity_log FOR INSERT WITH CHECK ((select auth.uid()) = user_id);


-- ============================================================
-- 2. FIX: unindexed_foreign_keys (Ensure full coverage)
-- ============================================================

-- activity_log: Upgrade partial index to full index for safe FK operations
DROP INDEX IF EXISTS idx_activity_user_id;
CREATE INDEX IF NOT EXISTS idx_activity_user_id ON public.activity_log(user_id);

-- recommendation_result (Conditional safety block)
DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'recommendation_result') THEN
        CREATE INDEX IF NOT EXISTS idx_recommendation_result_movie_id ON public.recommendation_result (movie_id);
        CREATE INDEX IF NOT EXISTS idx_recommendation_result_user_profile_id ON public.recommendation_result (user_profile_id);
    END IF;
END $$;

-- ============================================================
-- 3. FINAL VERIFICATION
-- ============================================================
-- - Fully qualified table names (public.*)
-- - DROP POLICY IF EXISTS before specific CREATE POLICY
-- - Idempotent
