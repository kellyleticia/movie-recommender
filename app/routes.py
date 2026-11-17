from fastapi import APIRouter, HTTPException, Query

from app.catalog import MovieCatalog
from app.content_recommender import recommend_by_content
from app.models import (
    ContentRecommendationsResponse,
    GenreRecommendationsResponse,
    Movie,
)
from app.recommender import recommend_movies
from app.similarity import SimilarityMatrix


def create_router(
    catalog: MovieCatalog,
    movies: list[Movie],
    similarities: SimilarityMatrix,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/movies", response_model=list[Movie])
    def list_movies() -> list[Movie]:
        return catalog.list_movies()

    @router.get(
        "/movies/{movie_id}/recommendations",
        response_model=GenreRecommendationsResponse,
    )
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
                    "overview": recommendation.movie.overview,
                    "shared_genres_count": recommendation.shared_genres_count,
                }
                for recommendation in recommendations
            ],
        }

    @router.get(
        "/movies/{movie_id}/content-recommendations",
        response_model=ContentRecommendationsResponse,
    )
    def movie_content_recommendations(
        movie_id: str, limit: int = Query(10, ge=1, le=50)
    ) -> ContentRecommendationsResponse:
        reference_movie = catalog.get_movie(movie_id)
        if reference_movie is None:
            raise HTTPException(status_code=404, detail="Movie not found")

        recommendations = recommend_by_content(
            reference_movie, similarities, movies, limit=limit
        )
        return ContentRecommendationsResponse(
            reference_movie_id=reference_movie.id,
            recommendations=[
                {
                    "id": recommendation.movie.id,
                    "title": recommendation.movie.title,
                    "genres": recommendation.movie.genres,
                    "overview": recommendation.movie.overview,
                    "similarity_score": recommendation.similarity_score,
                }
                for recommendation in recommendations
            ],
        )

    return router
