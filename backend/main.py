from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .inference import InferenceController
from .memory import MemoryManager
from .routers import recommendations, actors
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

app = FastAPI(
    title="Movie Recommendation AI",
    description="Python-native backend with Memory Orchestration",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
inference_controller = InferenceController()
memory_manager = MemoryManager()

@app.on_event("startup")
async def startup_event():
    print("🚀 System Startup: Initializing Memory & AI...")
    success = inference_controller.load_model()
    if not success:
        print("⚠️ Warning: Model failed to load. Inference will be disabled.")
    
    # Attach state to app for dependency injection
    app.state.inference = inference_controller
    app.state.memory = memory_manager
    print("✅ System Ready.")

# Routers
app.include_router(recommendations.router, prefix="/api", tags=["Recommendations"])
app.include_router(actors.router, prefix="/api/actors", tags=["Actors"])

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": inference_controller.recommender is not None,
        "memory_active": True
    }

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
