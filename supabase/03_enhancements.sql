-- =========================================================================
-- INCREMENTAL SCHEMA ENHANCEMENTS
-- Run this AFTER you've already created the basic schema
-- =========================================================================
-- This file adds production enhancements to existing tables:
-- - New columns
-- - CHECK constraints
-- - Additional indexes
-- - Helper functions
-- - Materialized views
-- =========================================================================

-- Enable additional extension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =========================================================================
-- PROFILES ENHANCEMENTS
-- =========================================================================

-- Add missing columns (safe - only adds if doesn't exist)
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='profiles' AND column_name='display_name') THEN
    ALTER TABLE profiles ADD COLUMN display_name TEXT;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='profiles' AND column_name='bio') THEN
    ALTER TABLE profiles ADD COLUMN bio TEXT;
  END IF;
END $$;

-- Add constraints
DO $$
BEGIN
  -- username_length constraint
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='username_length') THEN
    ALTER TABLE profiles 
      ADD CONSTRAINT username_length 
      CHECK (char_length(username) >= 3 AND char_length(username) <= 30);
  END IF;
  
  -- username_format constraint
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='username_format') THEN
    ALTER TABLE profiles 
      ADD CONSTRAINT username_format 
      CHECK (username ~ '^[a-zA-Z0-9_]+$');
  END IF;
  
  -- bio_length constraint
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='bio_length') THEN
    ALTER TABLE profiles 
      ADD CONSTRAINT bio_length 
      CHECK (char_length(bio) <= 500);
  END IF;
END $$;

-- Add case-insensitive username index (with pre-check for duplicates)
DO $$
BEGIN
  -- Check for case-insensitive duplicates before creating index
  IF EXISTS (
    SELECT LOWER(username), COUNT(*)
    FROM profiles
    GROUP BY LOWER(username)
    HAVING COUNT(*) > 1
  ) THEN
    RAISE EXCEPTION 'Case-insensitive duplicate usernames found. Please resolve before creating unique index.';
  ELSE
    DROP INDEX IF EXISTS idx_profiles_username_lower;
    CREATE UNIQUE INDEX idx_profiles_username_lower ON profiles(LOWER(username));
  END IF;
END $$;

-- =========================================================================
-- WATCHLIST ENHANCEMENTS
-- =========================================================================

-- Add missing columns
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='watchlist' AND column_name='notes') THEN
    ALTER TABLE watchlist ADD COLUMN notes TEXT;
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='watchlist' AND column_name='is_watched') THEN
    ALTER TABLE watchlist ADD COLUMN is_watched BOOLEAN DEFAULT FALSE;
  END IF;
END $$;

-- Add constraints
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='movie_id_format') THEN
    ALTER TABLE watchlist 
      ADD CONSTRAINT movie_id_format 
      CHECK (movie_id ~ '^\d+$');
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='notes_length') THEN
    ALTER TABLE watchlist 
      ADD CONSTRAINT notes_length 
      CHECK (char_length(notes) <= 1000);
  END IF;
END $$;

-- Add composite indexes
DROP INDEX IF EXISTS idx_watchlist_user_movie;
CREATE INDEX idx_watchlist_user_movie ON watchlist(user_id, movie_id);

DROP INDEX IF EXISTS idx_watchlist_is_watched;
CREATE INDEX idx_watchlist_is_watched ON watchlist(user_id, is_watched) WHERE is_watched = TRUE;

-- =========================================================================
-- ACTIVITY_LOG ENHANCEMENTS
-- =========================================================================

-- Add missing column
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='activity_log' AND column_name='session_id') THEN
    ALTER TABLE activity_log ADD COLUMN session_id UUID;
  END IF;
END $$;

-- Add constraints
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='action_not_empty') THEN
    ALTER TABLE activity_log 
      ADD CONSTRAINT action_not_empty 
      CHECK (char_length(action) > 0);
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.constraint_column_usage 
                 WHERE constraint_name='movie_id_format_activity') THEN
    ALTER TABLE activity_log 
      ADD CONSTRAINT movie_id_format_activity 
      CHECK (movie_id IS NULL OR movie_id ~ '^\d+$');
  END IF;
END $$;

-- Add partial indexes (more efficient)
DROP INDEX IF EXISTS idx_activity_session;
CREATE INDEX idx_activity_session ON activity_log(session_id) WHERE session_id IS NOT NULL;

DROP INDEX IF EXISTS idx_activity_movie_id;
CREATE INDEX idx_activity_movie_id ON activity_log(movie_id) WHERE movie_id IS NOT NULL;

DROP INDEX IF EXISTS idx_activity_user_action;
CREATE INDEX idx_activity_user_action ON activity_log(user_id, action) WHERE user_id IS NOT NULL;

-- Convert existing index to partial
DROP INDEX IF EXISTS idx_activity_user_id;
CREATE INDEX idx_activity_user_id ON activity_log(user_id) WHERE user_id IS NOT NULL;

-- =========================================================================
-- PREFERENCES ENHANCEMENTS
-- =========================================================================

-- Add missing columns
DO $$ 
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='preferences' AND column_name='exclude_genres') THEN
    ALTER TABLE preferences ADD COLUMN exclude_genres TEXT[] DEFAULT ARRAY[]::TEXT[];
  END IF;
  
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                 WHERE table_name='preferences' AND column_name='show_adult_content') THEN
    ALTER TABLE preferences ADD COLUMN show_adult_content BOOLEAN DEFAULT FALSE;
  END IF;
END $$;

-- Add GIN indexes for array columns
DROP INDEX IF EXISTS idx_preferences_favorite_genres;
CREATE INDEX idx_preferences_favorite_genres ON preferences USING GIN(favorite_genres);

DROP INDEX IF EXISTS idx_preferences_preferred_languages;
CREATE INDEX idx_preferences_preferred_languages ON preferences USING GIN(preferred_languages);

-- =========================================================================
-- UPDATE TRIGGER FUNCTION
-- =========================================================================
-- Function: Auto-create profile & preferences on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  -- Create profile with fallback username
  INSERT INTO public.profiles (id, username, avatar_url)
  VALUES (
    NEW.id,
    COALESCE(
      NEW.raw_user_meta_data->>'username',
      'user_' || substr(NEW.id::text, 1, 8)
    ),
    NEW.raw_user_meta_data->>'avatar_url'
  )
  ON CONFLICT (id) DO NOTHING;  -- Prevent duplicate key errors
  
  -- Create default preferences
  INSERT INTO public.preferences (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;  -- Prevent duplicate key errors
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public, pg_temp;

-- =========================================================================
-- NEW HELPER FUNCTIONS
-- =========================================================================

-- Function: Get watchlist count
CREATE OR REPLACE FUNCTION get_watchlist_count(uid UUID)
RETURNS INTEGER AS $$
  SELECT COUNT(*)::INTEGER FROM watchlist WHERE user_id = uid;
$$ LANGUAGE SQL STABLE SECURITY DEFINER
SET search_path = public, pg_temp;

COMMENT ON FUNCTION get_watchlist_count IS 'Returns total watchlist items for a user';

-- =========================================================================
-- MATERIALIZED VIEW: USER FEATURES
-- =========================================================================

DROP MATERIALIZED VIEW IF EXISTS user_features CASCADE;

CREATE MATERIALIZED VIEW user_features AS
SELECT 
  user_id,
  COUNT(*) FILTER (WHERE action = 'view') AS view_count,
  COUNT(*) FILTER (WHERE action = 'add_watchlist') AS save_count,
  COUNT(*) FILTER (WHERE action = 'search') AS search_count,
  MAX(created_at) AS last_active_at
FROM activity_log
WHERE user_id IS NOT NULL
GROUP BY user_id;

CREATE UNIQUE INDEX idx_user_features_user_id ON user_features(user_id);

COMMENT ON MATERIALIZED VIEW user_features IS 'Pre-computed user features for ML (refresh daily with: REFRESH MATERIALIZED VIEW user_features;)';

-- =========================================================================
-- ANALYTICS HELPER VIEWS
-- =========================================================================

DROP VIEW IF EXISTS recent_watchlist_activity CASCADE;

CREATE OR REPLACE VIEW recent_watchlist_activity AS
SELECT 
  w.user_id,
  w.movie_id,
  w.added_at,
  EXTRACT(EPOCH FROM (NOW() - w.added_at))/3600 AS hours_since_added
FROM watchlist w
WHERE w.added_at >= NOW() - INTERVAL '7 days'
ORDER BY w.added_at DESC;

COMMENT ON VIEW recent_watchlist_activity IS 'Recent watchlist activity (last 7 days) for trending analysis';

-- =========================================================================
-- ENHANCEMENTS COMPLETE
-- =========================================================================
-- 
-- Summary of changes applied:
-- ✅ Added display_name, bio to profiles
-- ✅ Added notes, is_watched to watchlist
-- ✅ Added session_id to activity_log
-- ✅ Added exclude_genres, show_adult_content to preferences
-- ✅ Added CHECK constraints for data validation
-- ✅ Added composite and partial indexes
-- ✅ Added GIN indexes for array searches
-- ✅ Created helper functions
-- ✅ Created materialized view for ML features
-- ✅ Created analytics helper views
--
-- Next steps:
-- 1. Verify changes: SELECT * FROM information_schema.columns WHERE table_name='profiles';
-- 2. Test indexing: EXPLAIN ANALYZE SELECT * FROM watchlist WHERE user_id = 'some-uuid';
-- 3. Refresh materialized view: REFRESH MATERIALIZED VIEW user_features;
