-- 04_enterprise_schema.sql
-- Security, Performance, and Advanced Views (Step 4 of 4)

-- ============================================================
-- 1. ROW LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;

-- Profiles Policies
CREATE POLICY "Public profiles are viewable by everyone" 
  ON profiles FOR SELECT USING (true);
  
CREATE POLICY "Users can insert their own profile" 
  ON profiles FOR INSERT WITH CHECK (auth.uid() = id);
  
CREATE POLICY "Users can update own profile" 
  ON profiles FOR UPDATE USING (auth.uid() = id);

-- Preferences Policies
CREATE POLICY "Users can view own preferences" 
  ON preferences FOR SELECT USING (auth.uid() = user_id);
  
CREATE POLICY "Users can insert own preferences" 
  ON preferences FOR INSERT WITH CHECK (auth.uid() = user_id);
  
CREATE POLICY "Users can update own preferences" 
  ON preferences FOR UPDATE USING (auth.uid() = user_id);

-- Watchlist Policies
CREATE POLICY "Users can view own watchlist" 
  ON watchlist FOR SELECT USING (auth.uid() = user_id);
  
CREATE POLICY "Users can insert own watchlist" 
  ON watchlist FOR INSERT WITH CHECK (auth.uid() = user_id);
  
CREATE POLICY "Users can update own watchlist" 
  ON watchlist FOR UPDATE USING (auth.uid() = user_id);
  
CREATE POLICY "Users can delete own watchlist" 
  ON watchlist FOR DELETE USING (auth.uid() = user_id);

-- Activity Log Policies
CREATE POLICY "Users can view own activity" 
  ON activity_log FOR SELECT USING (auth.uid() = user_id);
  
CREATE POLICY "Users can insert own activity" 
  ON activity_log FOR INSERT WITH CHECK (auth.uid() = user_id);
  



-- ============================================================
-- 2. INDEXING STRATEGY
-- ============================================================

-- Watchlist Indexes
CREATE INDEX IF NOT EXISTS idx_watchlist_user_id ON watchlist(user_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_added_at ON watchlist(added_at DESC);
CREATE INDEX IF NOT EXISTS idx_watchlist_user_movie ON watchlist(user_id, movie_id);
CREATE INDEX IF NOT EXISTS idx_watchlist_is_watched ON watchlist(user_id, is_watched) WHERE is_watched = TRUE;

-- Preferences Indexes
CREATE INDEX IF NOT EXISTS idx_preferences_user_id ON preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_preferences_favorite_genres ON preferences USING GIN(favorite_genres);
CREATE INDEX IF NOT EXISTS idx_preferences_preferred_languages ON preferences USING GIN(preferred_languages);

-- Activity Log Indexes
CREATE INDEX IF NOT EXISTS idx_activity_user_id ON activity_log(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_activity_created_at ON activity_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_action ON activity_log(action);
CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id) WHERE session_id IS NOT NULL;


-- ============================================================
-- 3. ANALYTICS & ML VIEWS
-- ============================================================

-- Materialized View for User Features (ML)
CREATE MATERIALIZED VIEW IF NOT EXISTS user_features AS
SELECT 
  al.user_id,
  ARRAY_AGG(DISTINCT g) FILTER (WHERE g IS NOT NULL) AS favorite_genres,
  COUNT(*) FILTER (WHERE al.action = 'view') AS view_count,
  COUNT(*) FILTER (WHERE al.action = 'add_watchlist') AS save_count,
  COUNT(*) FILTER (WHERE al.action = 'search') AS search_count,
  MAX(al.created_at) AS last_active_at
FROM activity_log al
LEFT JOIN preferences p ON al.user_id = p.user_id
LEFT JOIN LATERAL UNNEST(p.favorite_genres) AS g ON TRUE
WHERE al.user_id IS NOT NULL
GROUP BY al.user_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_features_user_id ON user_features(user_id);

-- Real-time View for Recent Activity
CREATE OR REPLACE VIEW recent_watchlist_activity AS
SELECT 
  w.user_id,
  w.movie_id,
  w.added_at,
  EXTRACT(EPOCH FROM (NOW() - w.added_at))/3600 AS hours_since_added
FROM watchlist w
WHERE w.added_at >= NOW() - INTERVAL '7 days'
ORDER BY w.added_at DESC;
