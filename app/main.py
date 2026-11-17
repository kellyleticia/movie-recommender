from fastapi import FastAPI

from app.catalog import MovieCatalog
from app.models import Movie
from app.routes import create_router
from app.similarity import compute_similarity
from app.text_features import build_tfidf_matrix


def create_app(catalog: MovieCatalog | None = None) -> FastAPI:
    if catalog is None:
        catalog = MovieCatalog(
            [
                Movie(
                    id="arrival",
                    title="Arrival",
                    genres=["Drama", "Sci-Fi"],
                    overview=(
                        "Uma linguista tenta compreender a chegada de naves "
                        "alienígenas."
                    ),
                ),
                Movie(
                    id="contact",
                    title="Contact",
                    genres=["Drama", "Sci-Fi"],
                    overview=(
                        "Uma astrônoma recebe um sinal que pode mudar a humanidade."
                    ),
                ),
                Movie(
                    id="amelie",
                    title="Amélie",
                    genres=["Comedy", "Romance"],
                    overview=(
                        "Uma jovem parisiense transforma a vida das pessoas "
                        "ao seu redor."
                    ),
                ),
            ]
        )

    movies = catalog.list_movies()
    representation = build_tfidf_matrix(movies)
    similarities = compute_similarity(representation)

    app = FastAPI(title="Movie Recommender")
    app.include_router(create_router(catalog, movies, similarities))
    return app


app = create_app()
