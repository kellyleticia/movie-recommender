from typing import Annotated

from pydantic import BaseModel, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Movie(BaseModel):
    id: NonEmptyString
    title: NonEmptyString
    genres: list[NonEmptyString]

    @property
    def normalized_genres(self) -> frozenset[str]:
        """Return genre names in a form suitable for case-insensitive comparison."""
        return frozenset(genre.casefold() for genre in self.genres)
