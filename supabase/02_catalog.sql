-- 02_catalog.sql
-- User's saved interactions and preferences (Step 2 of 4)

-- 1. Preferences Table
-- Stores user customization settings
CREATE TABLE IF NOT EXISTS public.preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    favorite_genres TEXT[] DEFAULT ARRAY[]::TEXT[],
    preferred_languages TEXT[] DEFAULT ARRAY['en', 'hi'],
    exclude_genres TEXT[] DEFAULT ARRAY[]::TEXT[],
    show_adult_content BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Auto-update updated_at for preferences
CREATE TRIGGER preferences_updated_at
    BEFORE UPDATE ON public.preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- 2. Watchlist Table
-- Stores user's saved movies
CREATE TABLE IF NOT EXISTS public.watchlist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id TEXT NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    notes TEXT,
    is_watched BOOLEAN DEFAULT FALSE,
    
    -- Constraints
    UNIQUE(user_id, movie_id),
    CONSTRAINT movie_id_format CHECK (movie_id ~ '^\d+$'),
    CONSTRAINT notes_length CHECK (char_length(notes) <= 1000)
);

-- 3. Trigger to initialize preferences for new users
CREATE OR REPLACE FUNCTION public.handle_new_user_prefs()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM set_config('search_path', 'public, pg_temp', true);
    INSERT INTO public.preferences (user_id) VALUES (NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created_prefs
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user_prefs();
