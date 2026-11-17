import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.catalog import MovieCatalog
from app.main import create_app
from app.models import Movie
from app.similarity import SimilarityMatrix
from app.text_features import TfidfRepresentation


def make_catalog() -> MovieCatalog:
    return MovieCatalog(
        [
            Movie(
                id="reference",
                title="Reference",
                genres=["Drama"],
                overview="Reference synopsis.",
            ),
            Movie(id="candidate-1", title="Candidate 1", genres=["Drama"]),
            Movie(id="candidate-2", title="Candidate 2", genres=["Drama"]),
            Movie(id="candidate-3", title="Candidate 3", genres=["Drama"]),
            Movie(id="unrelated", title="Unrelated", genres=["Comedy"]),
            Movie(id="no-genres", title="No Genres", genres=[]),
        ]
    )


def make_client(catalog: MovieCatalog | None = None) -> TestClient:
    return TestClient(create_app(catalog if catalog is not None else make_catalog()))


def make_content_catalog() -> MovieCatalog:
    return MovieCatalog(
        [
            Movie(
                id="reference",
                title="Reference",
                genres=["Sci-Fi"],
                overview="A crew explores a distant alien planet",
            ),
            Movie(
                id="partial",
                title="Partial match",
                genres=["Drama"],
                overview="A distant planet appears",
            ),
            Movie(
                id="related",
                title="Related match",
                genres=["Adventure"],
                overview=(
                    "A crew explores a distant alien planet with astronauts aboard"
                ),
            ),
            Movie(
                id="unrelated",
                title="Unrelated",
                genres=["Comedy"],
                overview="A chef prepares a meal in a restaurant",
            ),
            Movie(id="no-overview", title="No overview", genres=["Mystery"]),
        ]
    )


def test_health_check() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lists_movies() -> None:
    catalog = make_catalog()
    response = make_client(catalog).get("/movies")

    assert response.status_code == 200
    assert response.json() == [movie.model_dump() for movie in catalog.list_movies()]
    assert [movie["id"] for movie in response.json()] == [
        "reference",
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "unrelated",
        "no-genres",
    ]
    assert response.json()[0]["overview"] == "Reference synopsis."
    assert response.json()[1]["overview"] is None


def test_movies_openapi_schema_references_movie_model() -> None:
    schema = create_app(make_catalog()).openapi()

    response_schema = schema["paths"]["/movies"]["get"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]

    assert response_schema["type"] == "array"
    assert response_schema["items"] == {"$ref": "#/components/schemas/Movie"}


def test_genre_recommendations_openapi_schema_has_response_fields() -> None:
    schema = create_app(make_catalog()).openapi()

    response_schema = schema["paths"][
        "/movies/{movie_id}/recommendations"
    ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]

    assert response_schema == {
        "$ref": "#/components/schemas/GenreRecommendationsResponse"
    }
    components = schema["components"]["schemas"]
    assert set(components["GenreRecommendationsResponse"]["properties"]) == {
        "movie_id",
        "recommendations",
    }
    recommendation_schema = components["GenreRecommendationItem"]["properties"]
    assert set(recommendation_schema) == {
        "id",
        "title",
        "genres",
        "overview",
        "shared_genres_count",
    }


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
                "overview": None,
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


def test_returns_content_recommendations_in_similarity_order() -> None:
    response = make_client(make_content_catalog()).get(
        "/movies/reference/content-recommendations"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reference_movie_id"] == "reference"
    assert [movie["id"] for movie in body["recommendations"]] == [
        "related",
        "partial",
    ]
    first_recommendation = body["recommendations"][0]
    assert first_recommendation["id"] == "related"
    assert first_recommendation["title"] == "Related match"
    assert first_recommendation["genres"] == ["Adventure"]
    assert first_recommendation["overview"] == (
        "A crew explores a distant alien planet with astronauts aboard"
    )
    assert 0 < first_recommendation["similarity_score"] <= 1
    assert body["recommendations"][0]["similarity_score"] > body[
        "recommendations"
    ][1]["similarity_score"]


def test_content_recommendations_accept_limit() -> None:
    response = make_client(make_content_catalog()).get(
        "/movies/reference/content-recommendations?limit=1"
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["recommendations"]] == ["related"]


def test_content_recommendations_return_not_found_for_unknown_movie() -> None:
    response = make_client(make_content_catalog()).get(
        "/movies/missing/content-recommendations"
    )

    assert response.status_code == 404


def test_content_recommendations_return_empty_for_no_text_matches() -> None:
    catalog = MovieCatalog(
        [
            Movie(
                id="reference", title="Reference", genres=[], overview="space travel"
            ),
            Movie(
                id="candidate", title="Candidate", genres=[], overview="garden flowers"
            ),
        ]
    )

    response = make_client(catalog).get(
        "/movies/reference/content-recommendations"
    )

    assert response.status_code == 200
    assert response.json() == {
        "reference_movie_id": "reference",
        "recommendations": [],
    }


def test_content_recommendations_handle_reference_without_overview() -> None:
    catalog = MovieCatalog(
        [
            Movie(id="reference", title="Reference", genres=[]),
            Movie(
                id="candidate", title="Candidate", genres=[], overview="space travel"
            ),
        ]
    )

    response = make_client(catalog).get(
        "/movies/reference/content-recommendations"
    )

    assert response.status_code == 200
    assert response.json() == {
        "reference_movie_id": "reference",
        "recommendations": [],
    }


def test_content_structures_are_computed_once_per_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = make_content_catalog()
    original_build = main_module.build_tfidf_matrix
    original_similarity = main_module.compute_similarity
    calls = {"tfidf": 0, "similarity": 0}

    def count_build(movies: list[Movie]) -> TfidfRepresentation:
        calls["tfidf"] += 1
        return original_build(movies)

    def count_similarity(representation: TfidfRepresentation) -> SimilarityMatrix:
        calls["similarity"] += 1
        return original_similarity(representation)

    monkeypatch.setattr(main_module, "build_tfidf_matrix", count_build)
    monkeypatch.setattr(main_module, "compute_similarity", count_similarity)
    client = TestClient(main_module.create_app(catalog))

    client.get("/movies/reference/content-recommendations")
    client.get("/movies/reference/content-recommendations?limit=1")

    assert calls == {"tfidf": 1, "similarity": 1}
