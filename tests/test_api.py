from fastapi.testclient import TestClient

from app.catalog import MovieCatalog
from app.main import create_app
from app.models import Movie


def make_catalog() -> MovieCatalog:
    return MovieCatalog(
        [
            Movie(id="reference", title="Reference", genres=["Drama"]),
            Movie(id="candidate-1", title="Candidate 1", genres=["Drama"]),
            Movie(id="candidate-2", title="Candidate 2", genres=["Drama"]),
            Movie(id="candidate-3", title="Candidate 3", genres=["Drama"]),
            Movie(id="unrelated", title="Unrelated", genres=["Comedy"]),
            Movie(id="no-genres", title="No Genres", genres=[]),
        ]
    )


def make_client(catalog: MovieCatalog | None = None) -> TestClient:
    return TestClient(create_app(catalog if catalog is not None else make_catalog()))


def test_health_check() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lists_movies() -> None:
    response = make_client().get("/movies")

    assert response.status_code == 200
    assert [movie["id"] for movie in response.json()] == [
        "reference",
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "unrelated",
        "no-genres",
    ]


def test_returns_recommendations_successfully() -> None:
    response = make_client().get("/movies/reference/recommendations")

    assert response.status_code == 200
    assert response.json() == {
        "movie_id": "reference",
        "recommendations": [
            {
                "id": f"candidate-{index}",
                "title": f"Candidate {index}",
                "genres": ["Drama"],
                "shared_genres_count": 1,
            }
            for index in range(1, 4)
        ],
    }


def test_uses_default_limit_of_ten() -> None:
    catalog = MovieCatalog(
        [Movie(id="reference", title="Reference", genres=["Drama"])]
        + [
            Movie(
                id=f"candidate-{index:02}",
                title=f"Candidate {index:02}",
                genres=["Drama"],
            )
            for index in range(12)
        ]
    )

    response = make_client(catalog).get("/movies/reference/recommendations")

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 10


def test_accepts_custom_limit() -> None:
    response = make_client().get("/movies/reference/recommendations?limit=1")

    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 1


def test_rejects_limit_below_minimum() -> None:
    response = make_client().get("/movies/reference/recommendations?limit=0")

    assert response.status_code == 422


def test_rejects_limit_above_maximum() -> None:
    response = make_client().get("/movies/reference/recommendations?limit=51")

    assert response.status_code == 422


def test_returns_not_found_for_unknown_movie() -> None:
    response = make_client().get("/movies/missing/recommendations")

    assert response.status_code == 404


def test_returns_empty_recommendations_when_no_matches_exist() -> None:
    response = make_client().get("/movies/unrelated/recommendations")

    assert response.status_code == 200
    assert response.json() == {"movie_id": "unrelated", "recommendations": []}
