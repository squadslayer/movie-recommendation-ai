-- 03_interactions.sql
-- Activity logging and event tracking (Step 3 of 4)

-- 1. Activity Log Table
-- Records all user actions for analytics and ML
CREATE TABLE IF NOT EXISTS public.activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, -- Nullable for anonymous logs
    action TEXT NOT NULL,
    movie_id TEXT,
    session_id UUID,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    
    -- Validation
    CONSTRAINT action_not_empty CHECK (char_length(action) > 0),
    CONSTRAINT movie_id_format_activity CHECK (movie_id IS NULL OR movie_id ~ '^\d+$')
);

-- Comments for documentation
COMMENT ON TABLE activity_log IS 'Complete user interaction log (ML training data)';
COMMENT ON COLUMN activity_log.action IS 'Event type: search, view, add_watchlist, remove_watchlist, filter, recommend_request';
COMMENT ON COLUMN activity_log.session_id IS 'Groups events by session for funnel analysis';
