from fastapi import APIRouter, HTTPException, Depends, Request, Query
from typing import Optional
from ..inference import InferenceController
from ..memory import MemoryManager
from src.utils.wikipedia_images import get_actor_image_simple, get_actor_info_simple

router = APIRouter()

def get_inference_controller(request: Request) -> InferenceController:
    return request.app.state.inference

@router.get("/{actor_name}/movies")
async def get_actor_movies(
    actor_name: str,
    limit: int = 20,
    controller: InferenceController = Depends(get_inference_controller)
):
    """Get all movies featuring a specific actor."""
    try:
        movies = controller.get_actor_movies(actor_name, limit)
        
        results = []
        for movie_id, rating in movies:
            info = controller.recommender.get_movie_info(movie_id)
            if info:
                poster_url = controller._get_poster_url(movie_id)
                results.append({
                    'id': movie_id,
                    'title': info['title'],
                    'year': info.get('year', 'N/A'),
                    'rating': float(rating),
                    'genre': info.get('genre', 'N/A'),
                    'poster_url': poster_url
                })
        
        return {
            'actor': actor_name,
            'total_movies': len(results),
            'movies': results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{actor_name}/profile")
async def get_actor_profile(
    actor_name: str,
    controller: InferenceController = Depends(get_inference_controller)
):
    """Get actor profile including statistics."""
    try:
        data = controller.get_actor_info(actor_name)
        if not data:
            raise HTTPException(status_code=404, detail="Actor not found in database")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{actor_name}/photo")
async def get_actor_photo(
    actor_name: str,
    source: str = "wikipedia"
):
    """Get actor photo from Wikipedia."""
    if source != "wikipedia":
        raise HTTPException(status_code=400, detail="Unsupported source")
        
    try:
        photo_url = get_actor_image_simple(actor_name)
        return {
            'actor': actor_name,
            'photo_url': photo_url,
            'source': 'wikipedia'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{actor_name}/info")
async def get_actor_wikipedia_info(
    actor_name: str,
    controller: InferenceController = Depends(get_inference_controller)
):
    """Get comprehensive actor info from Wikipedia + Local DB."""
    try:
        wiki_info = get_actor_info_simple(actor_name)
        
        # Enrich with local data
        local_data = controller.get_actor_info(actor_name)
        if local_data:
            top_movies = local_data.get('top_movies', [])[:5]
            for movie in top_movies:
                movie['poster_url'] = controller._get_poster_url(movie['movie_id'])
            
            wiki_info['filmography'] = {
                'total_movies': local_data.get('total_movies', 0),
                'average_rating': local_data.get('average_rating', 0),
                'top_movies': top_movies
            }
            
        return wiki_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
