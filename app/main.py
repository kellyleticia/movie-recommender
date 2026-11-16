from fastapi import FastAPI

from app.catalog import MovieCatalog
from app.models import Movie
from app.routes import create_router


def create_app(catalog: MovieCatalog | None = None) -> FastAPI:
    if catalog is None:
        catalog = MovieCatalog(
            [
                Movie(id="arrival", title="Arrival", genres=["Drama", "Sci-Fi"]),
                Movie(id="contact", title="Contact", genres=["Drama", "Sci-Fi"]),
                Movie(id="amelie", title="Amélie", genres=["Comedy", "Romance"]),
            ]
        )

    app = FastAPI(title="Movie Recommender")
    app.include_router(create_router(catalog))
    return app


app = create_app()
