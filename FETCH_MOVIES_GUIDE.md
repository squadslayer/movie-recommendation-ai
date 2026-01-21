# Fetch 90K Movies - Quick Guide

## What It Does
Fetches ~90K quality movies from TMDB to expand your dataset from 10K → 100K movies.

## Quality Filters
- ✅ Minimum 50 votes (popular)
- ✅ Minimum 5.0/10 rating (decent quality)  
- ✅ No adult content

## Columns Included
All 12 required columns:
`id`, `original_language`, `overview`, `release_date`, `title`, `popularity`, `vote_average`, `vote_count`, `genre`, `director`, `year`, `cast`

## How to Run

```bash
# Install dependency (if not already installed)
pip install aiohttp

# Run the fetcher
python fetch_movies_async.py
```

## Progress
- The script will log progress every page
- Checkpoints saved every 1,000 movies
- You can ctrl+c anytime - it will save progress
- Resume by running again (skips duplicates)

## Estimated Time
**7-8 hours** for ~90K movies (API rate limits apply)

## After Completion
1. Check your dataset: `wc -l data/movies_enriched.csv` (should be ~100K rows)
2. Retrain model: `python train_model.py`
3. Restart backend (it will auto-load new model)
