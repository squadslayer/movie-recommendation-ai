-- =========================================================================
-- PRODUCTION-GRADE SUPABASE SCHEMA
-- Movie Discovery Platform - Enhanced Version
-- =========================================================================
-- This schema includes all production enhancements:
-- - Data validation (CHECK constraints)
-- - Composite indexes for performance
-- - Session tracking
-- - ML-ready features
-- - Analytics optimization
-- =========================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =========================================================================
-- TABLE 1: PROFILES
-- =========================================================================
-- Extends auth.users with application profile data
-- Relationship: 1:1 with auth.users

CREATE TABLE profiles (
  -- Primary Key (FK to auth.users)
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Profile Data
  username TEXT UNIQUE NOT NULL,
  display_name TEXT,
  avatar_url TEXT,
  bio TEXT,
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  
  -- Constraints
  CONSTRAINT username_length CHECK (char_length(username) >= 3 AND char_length(username) <= 30),
  CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9_]+$'),
  CONSTRAINT bio_length CHECK (char_length(bio) <= 500)
);

-- Indexes
CREATE UNIQUE INDEX idx_profiles_username_lower ON profiles(LOWER(username));

-- Comments
COMMENT ON TABLE profiles IS 'User profiles extending Supabase auth.users (1:1)';
COMMENT ON COLUMN profiles.username IS 'Unique username (3-30 alphanumeric chars + underscore)';
COMMENT ON CONSTRAINT username_format ON profiles IS 'Alphanumeric + underscore only';

-- =========================================================================
-- TABLE 2: WATCHLIST
-- =========================================================================
-- User's saved movies ("My List" feature)
-- Relationship: N:1 with auth.users

CREATE TABLE watchlist (
  -- Primary Key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- Foreign Keys
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  movie_id TEXT NOT NULL,
  
  -- Metadata
  added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  notes TEXT,
  is_watched BOOLEAN DEFAULT FALSE,
  
  -- Constraints
  UNIQUE(user_id, movie_id),
  CONSTRAINT movie_id_format CHECK (movie_id ~ '^\d+$'),
  CONSTRAINT notes_length CHECK (char_length(notes) <= 1000)
);

-- Indexes
CREATE INDEX idx_watchlist_user_id ON watchlist(user_id);
CREATE INDEX idx_watchlist_added_at ON watchlist(added_at DESC);
CREATE INDEX idx_watchlist_user_movie ON watchlist(user_id, movie_id);
CREATE INDEX idx_watchlist_is_watched ON watchlist(user_id, is_watched) WHERE is_watched = TRUE;

-- Comments
COMMENT ON TABLE watchlist IS 'User saved movies + watched status (implicit feedback)';
COMMENT ON COLUMN watchlist.movie_id IS 'TMDB movie ID (numeric string)';
COMMENT ON COLUMN watchlist.is_watched IS 'Distinguishes intent (saved) from completion (watched)';
COMMENT ON COLUMN watchlist.added_at IS 'Critical for time-decay recommendation models';

-- =========================================================================
-- TABLE 3: ACTIVITY_LOG
-- =========================================================================
-- ALL user interactions (search, view, click, etc)
-- Relationship: N:1 with auth.users (nullable for anonymous)

CREATE TABLE activity_log (
  -- Primary Key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- Foreign Key (nullable for anonymous users)
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  
  -- Event Data
  action TEXT NOT NULL,
  movie_id TEXT,
  session_id UUID,
  
  -- Context
  metadata JSONB,
  
  -- Timestamp
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  
  -- Constraints
  CONSTRAINT action_not_empty CHECK (char_length(action) > 0),
  CONSTRAINT movie_id_format_activity CHECK (movie_id IS NULL OR movie_id ~ '^\d+$')
);

-- Indexes (partial indexes for better performance)
CREATE INDEX idx_activity_user_id ON activity_log(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_activity_created_at ON activity_log(created_at DESC);
CREATE INDEX idx_activity_action ON activity_log(action);
CREATE INDEX idx_activity_session ON activity_log(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX idx_activity_movie_id ON activity_log(movie_id) WHERE movie_id IS NOT NULL;
CREATE INDEX idx_activity_user_action ON activity_log(user_id, action) WHERE user_id IS NOT NULL;

-- Comments
COMMENT ON TABLE activity_log IS 'Complete user interaction log (ML training data)';
COMMENT ON COLUMN activity_log.action IS 'Event type: search, view, add_watchlist, remove_watchlist, filter, recommend_request';
COMMENT ON COLUMN activity_log.session_id IS 'Groups events by session for funnel analysis';
COMMENT ON COLUMN activity_log.metadata IS 'Flexible JSONB for context (search query, filters, referrer)';
COMMENT ON COLUMN activity_log.user_id IS 'Nullable to allow anonymous analytics (GDPR-compliant)';

-- =========================================================================
-- TABLE 4: PREFERENCES
-- =========================================================================
-- Explicit user preferences for content filtering
-- Relationship: 1:1 with auth.users

CREATE TABLE preferences (
  -- Primary Key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- Foreign Key (1:1 with user)
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Preferences
  favorite_genres TEXT[] DEFAULT ARRAY[]::TEXT[],
  preferred_languages TEXT[] DEFAULT ARRAY['en', 'hi'],
  exclude_genres TEXT[] DEFAULT ARRAY[]::TEXT[],
  
  -- Settings
  show_adult_content BOOLEAN DEFAULT FALSE,
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Indexes
CREATE INDEX idx_preferences_user_id ON preferences(user_id);
CREATE INDEX idx_preferences_favorite_genres ON preferences USING GIN(favorite_genres);
CREATE INDEX idx_preferences_preferred_languages ON preferences USING GIN(preferred_languages);

-- Comments
COMMENT ON TABLE preferences IS 'User explicit preferences (1:1 with user)';
COMMENT ON COLUMN preferences.favorite_genres IS 'Positive signals for recommendation';
COMMENT ON COLUMN preferences.exclude_genres IS 'Hard constraints (never recommend these)';
COMMENT ON COLUMN preferences.preferred_languages IS 'en=English/Hollywood, hi=Hindi/Bollywood';

-- =========================================================================
-- FUNCTIONS & TRIGGERS
-- =========================================================================

-- Function: Auto-update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER preferences_updated_at
  BEFORE UPDATE ON preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

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
  );
  
  -- Create default preferences
  INSERT INTO public.preferences (user_id)
  VALUES (NEW.id);
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION handle_new_user();

-- Function: Get watchlist count (helper)
CREATE OR REPLACE FUNCTION get_watchlist_count(uid UUID)
RETURNS INTEGER AS $$
  SELECT COUNT(*)::INTEGER FROM watchlist WHERE user_id = uid;
$$ LANGUAGE SQL STABLE SECURITY DEFINER;

COMMENT ON FUNCTION get_watchlist_count IS 'Helper function for dashboard stats';

-- =========================================================================
-- MATERIALIZED VIEW: USER FEATURES (ML/Analytics)
-- =========================================================================
-- Pre-computed user features for ML training
-- Refresh schedule: DAILY via cron job

CREATE MATERIALIZED VIEW user_features AS
SELECT 
  user_id,
  ARRAY_AGG(DISTINCT favorite_genres) FILTER (WHERE favorite_genres IS NOT NULL) AS favorite_genres,
  COUNT(*) FILTER (WHERE action = 'view') AS view_count,
  COUNT(*) FILTER (WHERE action = 'add_watchlist') AS save_count,
  COUNT(*) FILTER (WHERE action = 'search') AS search_count,
  MAX(created_at) AS last_active_at
FROM activity_log
WHERE user_id IS NOT NULL
GROUP BY user_id;

CREATE UNIQUE INDEX idx_user_features_user_id ON user_features(user_id);

COMMENT ON MATERIALIZED VIEW user_features IS 'Pre-computed user features for ML (refresh daily)';

-- =========================================================================
-- ANALYTICS HELPER VIEWS
-- =========================================================================

-- View: Recent watchlist additions (last 7 days)
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
-- SETUP COMPLETE
-- =========================================================================
-- Next step: Run 02_rls_policies.sql to secure your data
-- 
-- Performance notes:
-- - FK columns are explicitly indexed above for JOIN/lookup performance
-- - Partial indexes reduce index size by 30-50%
-- - GIN indexes enable fast array searches
-- - Materialized view MUST be refreshed manually/via cron
--
-- Refresh materialized view:
-- REFRESH MATERIALIZED VIEW user_features;
