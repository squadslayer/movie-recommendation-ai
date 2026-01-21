-- ============================================================
-- Row-Level Security (RLS) Policies
-- ============================================================
-- Ensures users can only access their own data
-- Run this AFTER 01_schema.sql
-- ============================================================

-- ============================================================
-- ENABLE RLS ON ALL TABLES
-- ============================================================

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlist ENABLE ROW LEVEL SECURITY;
ALTER TABLE activity_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE preferences ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- PROFILES POLICIES
-- ============================================================

-- Users can view their own profile
CREATE POLICY "Users can view own profile"
  ON profiles
  FOR SELECT
  USING (auth.uid() = id);

-- Users can update their own profile
CREATE POLICY "Users can update own profile"
  ON profiles
  FOR UPDATE
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- Users can insert their own profile
CREATE POLICY "Users can insert own profile"
  ON profiles
  FOR INSERT
  WITH CHECK (auth.uid() = id);

-- ============================================================
-- WATCHLIST POLICIES
-- ============================================================

-- Users can view their own watchlist
CREATE POLICY "Users can view own watchlist"
  ON watchlist
  FOR SELECT
  USING (auth.uid() = user_id);

-- Users can add to their own watchlist
CREATE POLICY "Users can insert to own watchlist"
  ON watchlist
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Users can remove from their own watchlist
CREATE POLICY "Users can delete from own watchlist"
  ON watchlist
  FOR DELETE
  USING (auth.uid() = user_id);

-- ============================================================
-- ACTIVITY LOG POLICIES
-- ============================================================

-- Users can view their own activity
CREATE POLICY "Users can view own activity"
  ON activity_log
  FOR SELECT
  USING (auth.uid() = user_id);

-- Users can insert their own activity
CREATE POLICY "Users can insert own activity"
  ON activity_log
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Allow anonymous activity logging (optional - for guest users)
CREATE POLICY "Allow anonymous activity logging"
  ON activity_log
  FOR INSERT
  WITH CHECK (user_id IS NULL);

-- ============================================================
-- PREFERENCES POLICIES
-- ============================================================

-- Users can view their own preferences
CREATE POLICY "Users can view own preferences"
  ON preferences
  FOR SELECT
  USING (auth.uid() = user_id);

-- Users can update their own preferences
CREATE POLICY "Users can update own preferences"
  ON preferences
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Users can insert their own preferences
CREATE POLICY "Users can insert own preferences"
  ON preferences
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- ============================================================
-- RLS SETUP COMPLETE
-- ============================================================
-- Your database is now secured!
-- Users can only access their own data.
