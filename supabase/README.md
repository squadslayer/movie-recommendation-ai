# Supabase Migration Guide

## Quick Start

You have **3 SQL files** to run in order:

### If Starting Fresh (New Supabase Project)

1. **`01_schema.sql`** - Create all tables
2. **`02_rls_policies.sql`** - Add security policies
3. **Skip `03_enhancements.sql`** (already included in 01)

### If You Already Ran Basic Schema ✅ (YOUR SITUATION)

Run **ONLY**:
- **`03_enhancements.sql`** - Add missing features

---

## How to Run in Supabase

1. Go to **Supabase Dashboard**
2. Click **SQL Editor**
3. Click **+ New Query**
4. Copy contents of `03_enhancements.sql`
5. Click **Run** (green play button)

---

## What `03_enhancements.sql` Adds

### New Columns
- `profiles.display_name` - Separate from username
- `profiles.bio` - User bio (max 500 chars)
- `watchlist.notes` - Personal notes on movies
- `watchlist.is_watched` - Track completion status
- `activity_log.session_id` - Session tracking
- `preferences.exclude_genres` - Negative preferences
- `preferences.show_adult_content` - Content filter

### Data Validation
- Username: 3-30 chars, alphanumeric + underscore
- Movie ID: Must be numeric
- Bio: Max 500 chars
- Notes: Max 1000 chars

### Performance Improvements
- **Composite indexes** (user_id + movie_id)
- **Partial indexes** (WHERE clauses reduce size 30-50%)
- **GIN indexes** (fast array searches on genres)
- **Case-insensitive** username search

### Analytics
- **Materialized View:** `user_features` (ML training data)
- **View:** `recent_watchlist_activity` (trending)
- **Function:** `get_watchlist_count(user_id)`

---

## Verification

After running, test with:

```sql
-- Check new columns exist
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'watchlist';

-- Should show: id, user_id, movie_id, added_at, notes, is_watched

-- Test indexes
EXPLAIN ANALYZE 
SELECT * FROM watchlist 
WHERE user_id = 'your-user-id';

-- Should use idx_watchlist_user_id
```

---

## Maintenance

### Refresh Materialized View (Run Daily)
```sql
REFRESH MATERIALIZED VIEW user_features;
```

Set up in Supabase Dashboard → Database → Cron Jobs:
```sql
-- Every day at 2 AM
SELECT cron.schedule(
  'refresh-user-features',
  '0 2 * * *',
  'REFRESH MATERIALIZED VIEW user_features'
);
```

---

## Troubleshooting

**Error: "relation already exists"**
- Safe to ignore if using `DROP ... IF EXISTS`

**Error: "column already exists"**
- Migration uses `IF NOT EXISTS` - safe to rerun

**Error: "function does not exist"**
- Run `01_schema.sql` first

---

## Next Steps After Migration

1. ✅ Run `03_enhancements.sql`
2. Get API keys from Supabase Dashboard → Settings → API
3. Add to `.env`:
   ```env
   REACT_APP_SUPABASE_URL=https://xxx.supabase.co
   REACT_APP_SUPABASE_ANON_KEY=your-anon-key
   SUPABASE_SERVICE_KEY=your-service-key  # Backend only
   ```
4. Install dependencies: `npm install @supabase/supabase-js`
5. Proceed with frontend/backend integration
