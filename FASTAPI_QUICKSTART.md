# FastAPI Backend# FastAPI Quick Start

## Running the FastAPI Backend

1.  **Navigate to project directory:**
    ```bash
    cd path/to/movie-recommender
    ```

2.  **Run the server:**
    ```bash
    # Start FastAPI server (port 8000)
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
    ```

## Test Endpoints

```powershell
# Health check
curl http://localhost:8000/api/health

# Get recommendations (returns trace_id)
curl -X POST http://localhost:8000/api/recommendations ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"test\", \"input_movies\": [\"Inception\"], \"limit\": 5}"

# Audit a trace (replace TRACE_ID)
curl http://localhost:8000/api/audit/TRACE_ID

# Actor info
curl http://localhost:8000/api/actors/Tom%20Cruise/info
```

## Interactive API Docs

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Memory Log Location

Traces are stored in: `data/memory_log.jsonl`

## Need Help?

See `docs/fastapi_walkthrough.md` for full documentation.
