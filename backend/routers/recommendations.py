from fastapi import APIRouter, HTTPException, Depends, Request
from ..schemas import RecommendationRequest, PredictionResponse, AuditResponse
from ..inference import InferenceController
from ..memory import MemoryManager
import time

router = APIRouter()

def get_inference_controller(request: Request) -> InferenceController:
    return request.app.state.inference

def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory

@router.post("/recommendations", response_model=PredictionResponse)
async def get_recommendations(
    req: RecommendationRequest,
    controller: InferenceController = Depends(get_inference_controller),
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Generate recommendations and log the trace.
    """
    # 1. Start Trace
    trace_id = memory.start_trace(req.dict())
    
    try:
        # 2. Run Inference
        # Get predictions
        result = controller.predict(
            input_movies=req.input_movies,
            limit=req.limit
        )
        
        # Check for error in result
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        # Log prediction
        memory.log_prediction(trace_id, result)
        
        # Return response
        return PredictionResponse(
            trace_id=trace_id,
            model_version=result["model_version"],
            recommendations=result["recommendations"],
            explanations=result.get("explanations", {})
        )
        
    except HTTPException:
        # Re-raise HTTPException to preserve client status codes (e.g., 404)
        raise
    except Exception as e:
        # Log error in future
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit/{trace_id}", response_model=AuditResponse)
async def audit_trace(
    trace_id: str,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Retrieve full execution trace for debugging/auditing.
    """
    trace_context = memory.get_trace(trace_id)
    if not trace_context:
        raise HTTPException(status_code=404, detail="Trace not found")
        
    return AuditResponse(
        trace_id=trace_id,
        request=trace_context.get("request"),
        prediction=trace_context.get("prediction"),
        external_calls=trace_context.get("external_calls", [])
    )
