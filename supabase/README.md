# Supabase Setup Guide

This folder contains SQL migration files to set up your movie discovery platform database.

## 📁 Files

1. **`01_schema.sql`** - Core database tables
2. **`02_rls_policies.sql`** - Row-Level Security policies
3. **`README.md`** - This file

---

## 🚀 Quick Start

### Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com)
2. Click "New Project"
3. Fill in:
   - Project name: `movie-discovery`
   - Database password: **Save this!**
   - Region: Choose nearest to you
4. Wait ~1 minute for setup

### Step 2: Get Your API Keys

After project creation:

1. Go to **Settings** → **API**
2. Copy these values to your `.env` file:

```env
# Supabase Configuration
REACT_APP_SUPABASE_URL=https://your-project.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_KEY=your-service-key-here  # For backend only
```

### Step 3: Run SQL Migrations

1. Go to **SQL Editor** in Supabase dashboard
2. Click "+ New Query"
3. Copy & paste contents of `01_schema.sql`
4. Click "Run"
5. Repeat for `02_rls_policies.sql`

---

## 📊 Database Schema

### Tables Created

#### `profiles`
Extends `auth.users` with additional profile data.

| Column | Type | Description |
|:---|:---|:---|
| id | UUID | References auth.users(id) |
| username | TEXT | Unique username |
| avatar_url | TEXT | Profile picture URL |
| created_at | TIMESTAMP | Account creation |

#### `watchlist`
User's saved movies ("My List" feature).

| Column | Type | Description |
|:---|:---|:---|
| id | UUID | Primary key |
| user_id | UUID | References auth.users(id) |
| movie_id | TEXT | TMDB movie ID |
| added_at | TIMESTAMP | When saved |

#### `activity_log`
Tracks user interactions for personalization.

| Column | Type | Description |
|:---|:---|:---|
| id | UUID | Primary key |
| user_id | UUID | References auth.users(id) |
| action | TEXT | Action type (search, view, etc.) |
| movie_id | TEXT | Related movie (if applicable) |
| metadata | JSONB | Additional context |
| created_at | TIMESTAMP | When action occurred |

#### `preferences`
User's explicit preferences.

| Column | Type | Description |
|:---|:---|:---|
| id | UUID | Primary key |
| user_id | UUID | References auth.users(id) |
| favorite_genres | TEXT[] | Array of genres |
| preferred_languages | TEXT[] | ['en', 'hi'] for English/Hindi |

---

## 🔐 Security (Row-Level Security)

All tables have RLS enabled. Users can only:
- **Read** their own data
- **Write** to their own data
- **Delete** their own data

Anonymous users can log activity (for analytics).

---

## 🧪 Testing the Setup

After running migrations, test in SQL Editor:

```sql
-- Check tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Verify RLS is enabled
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public';
```

Expected output: 4 tables with `rowsecurity = true`

---

## 🔄 Auto-Created Features

### Triggers
- **Auto-update `updated_at`** on profile/preference changes
- **Auto-create profile** when new user signs up

### Indexes
Optimized queries for:
- Watchlist lookups
- Activity log analytics
- User preference filtering

---

## 📝 Data Science Notes

### Implicit Feedback Signals

The `activity_log` table captures:
- **Searches** → User intent
- **Views** → Interest signals
- **Watchlist additions** → Strong positive signal
- **Watchlist removals** → Negative signal

This data can be used for:
- Collaborative filtering
- User embeddings
- Session-based recommendations
- A/B testing

### Example Analytics Queries

```sql
-- Most popular movies this week
SELECT movie_id, COUNT(*) as views
FROM activity_log
WHERE action = 'view'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY movie_id
ORDER BY views DESC
LIMIT 10;

-- User's favorite genres (from watchlist)
SELECT u.id, array_agg(DISTINCT movie.genre)
FROM profiles u
JOIN watchlist w ON w.user_id = u.id
-- Join with your movie data to get genres
GROUP BY u.id;
```

---

## ⚠️ Important Notes

1. **Never expose `SUPABASE_SERVICE_KEY`** in frontend code
2. Use **anon key** for frontend
3. Use **service key** only in backend (Python)
4. **RLS is mandatory** - don't disable it
5. **Test policies** before going to production

---

## 🔗 Next Steps

1. **Frontend**: Install `@supabase/supabase-js`
2. **Backend**: Install `supabase-py`
3. **Test auth flow**: Sign up → Login → Watchlist
4. **Connect to recommendation engine**: Use activity log for personalization

---

## 📚 Resources

- [Supabase Docs](https://supabase.com/docs)
- [RLS Guide](https://supabase.com/docs/guides/auth/row-level-security)
- [Supabase + React](https://supabase.com/docs/guides/getting-started/tutorials/with-react)
