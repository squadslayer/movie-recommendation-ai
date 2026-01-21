-- ============================================================
-- Movie Discovery Platform - Database Schema
-- ============================================================
-- This file creates the core tables for user authentication,
-- watchlists, activity logging, and personalization.
--
-- Run this in Supabase SQL Editor after creating your project
-- ============================================================

-- Enable UUID extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. USER PROFILES
-- ============================================================
-- Extends Supabase auth.users with additional profile data
--
CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username TEXT UNIQUE,
  avatar_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE profiles IS 'User profile data extending Supabase auth.users';
COMMENT ON COLUMN profiles.id IS 'References auth.users(id) - primary auth identifier';
COMMENT ON COLUMN profiles.username IS 'Unique username for display';

-- ============================================================
-- 2. WATCHLIST (My List)
-- ============================================================
-- Stores user's saved movies for later viewing
--
CREATE TABLE watchlist (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  movie_id TEXT NOT NULL,
  added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, movie_id)
);

COMMENT ON TABLE watchlist IS 'User saved movies (My List feature)';
COMMENT ON COLUMN watchlist.movie_id IS 'TMDB movie ID (stored as text for flexibility)';
COMMENT ON COLUMN watchlist.added_at IS 'Timestamp for recommendation decay analysis';

-- Indexes for performance
CREATE INDEX idx_watchlist_user_id ON watchlist(user_id);
CREATE INDEX idx_watchlist_added_at ON watchlist(added_at DESC);

-- ============================================================
-- 3. ACTIVITY LOG
-- ============================================================
-- Tracks all user interactions for personalization & analytics
--
CREATE TABLE activity_log (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  movie_id TEXT,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE activity_log IS 'User activity tracking for personalization';
COMMENT ON COLUMN activity_log.action IS 'Action type: search, view, add_watchlist, remove_watchlist, etc.';
COMMENT ON COLUMN activity_log.metadata IS 'Additional context (search query, filters, etc.)';

-- Indexes for analytics queries
CREATE INDEX idx_activity_user_id ON activity_log(user_id);
CREATE INDEX idx_activity_created_at ON activity_log(created_at DESC);
CREATE INDEX idx_activity_action ON activity_log(action);

-- ============================================================
-- 4. USER PREFERENCES
-- ============================================================
-- Stores user's explicit preferences for personalization
--
CREATE TABLE preferences (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
  favorite_genres TEXT[] DEFAULT ARRAY[]::TEXT[],
  preferred_languages TEXT[] DEFAULT ARRAY['en', 'hi'],
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE preferences IS 'User preferences for content filtering';
COMMENT ON COLUMN preferences.favorite_genres IS 'Array of preferred genres (Action, Drama, etc.)';
COMMENT ON COLUMN preferences.preferred_languages IS 'Language preferences (en=English/Hollywood, hi=Hindi/Bollywood)';

-- Index for lookups
CREATE INDEX idx_preferences_user_id ON preferences(user_id);

-- ============================================================
-- TRIGGERS
-- ============================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_preferences_updated_at
  BEFORE UPDATE ON preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- AUTO-CREATE PROFILE ON USER SIGNUP
-- ============================================================
-- Automatically create a profile when a new user signs up
--
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, username, avatar_url)
  VALUES (
    NEW.id,
    NEW.raw_user_meta_data->>'username',
    NEW.raw_user_meta_data->>'avatar_url'
  );
  
  INSERT INTO public.preferences (user_id)
  VALUES (NEW.id);
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- SETUP COMPLETE
-- ============================================================
-- Next step: Run 02_rls_policies.sql to secure your data
