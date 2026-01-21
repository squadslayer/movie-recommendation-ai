from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class RecommendationRequest(BaseModel):
    user_id: Optional[str] = "guest"
    input_movies: List[str] = Field(..., description="List of movie IDs or titles")
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None

class MovieResponse(BaseModel):
    id: str
    title: str
    year: Optional[str] = None
    rating: float
    genre: Optional[str] = None
    director: Optional[str] = None
    score: float
    poster_url: Optional[str] = None
    overview: Optional[str] = None

class PredictionResponse(BaseModel):
    trace_id: str
    model_version: str
    recommendations: Dict[str, List[MovieResponse]]
    explanations: List[str]

class AuditResponse(BaseModel):
    trace_id: str
    request: Optional[Dict[str, Any]]
    prediction: Optional[Dict[str, Any]]
    external_calls: List[Dict[str, Any]]
