from fastapi import APIRouter, HTTPException, Query

from app.catalog import MovieCatalog
from app.recommender import recommend_movies


def create_router(catalog: MovieCatalog) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/movies")
    def list_movies():
        return catalog.list_movies()

    @router.get("/movies/{movie_id}/recommendations")
    def movie_recommendations(movie_id: str, limit: int = Query(10, ge=1, le=50)):
        reference_movie = catalog.get_movie(movie_id)
        if reference_movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")

        recommendations = recommend_movies(
            reference_movie, catalog.list_movies(), limit=limit
        )
        return {
            "movie_id": reference_movie.id,
            "recommendations": [
                {
                    "id": recommendation.movie.id,
                    "title": recommendation.movie.title,
                    "genres": recommendation.movie.genres,
                    "shared_genres_count": recommendation.shared_genres_count,
                }
                for recommendation in recommendations
            ],
        }

    return router
