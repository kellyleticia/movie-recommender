from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Movie(BaseModel):
    id: NonEmptyString
    title: NonEmptyString
    genres: list[NonEmptyString]
    overview: NonEmptyString | None = None

    @property
    def normalized_genres(self) -> frozenset[str]:
        """Return genre names in a form suitable for case-insensitive comparison."""
        return frozenset(genre.casefold() for genre in self.genres)


class ContentRecommendationItem(BaseModel):
    id: NonEmptyString
    title: NonEmptyString
    genres: list[NonEmptyString]
    overview: NonEmptyString | None
    similarity_score: float = Field(ge=0.0, le=1.0)


class ContentRecommendationsResponse(BaseModel):
    reference_movie_id: NonEmptyString
    recommendations: list[ContentRecommendationItem]


class GenreRecommendationItem(Movie):
    shared_genres_count: int


class GenreRecommendationsResponse(BaseModel):
    movie_id: NonEmptyString
    recommendations: list[GenreRecommendationItem]
