from app.models import Movie
from app.recommender import recommend_movies


def make_movie(movie_id: str, title: str, genres: list[str]) -> Movie:
    return Movie(id=movie_id, title=title, genres=genres)


def test_recommends_movie_with_one_shared_genre() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    candidate = make_movie("candidate", "Candidate", ["Drama", "Comedy"])

    result = recommend_movies(reference, [candidate])

    assert [(item.movie, item.shared_genres_count) for item in result] == [
        (candidate, 1)
    ]


def test_counts_multiple_shared_genres() -> None:
    reference = make_movie("ref", "Reference", ["Drama", "Comedy", "Drama"])
    candidate = make_movie("candidate", "Candidate", ["Comedy", "Drama"])

    result = recommend_movies(reference, [candidate])

    assert result[0].shared_genres_count == 2


def test_ignores_movies_without_shared_genres() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    candidate = make_movie("candidate", "Candidate", ["Comedy"])

    assert recommend_movies(reference, [candidate]) == []


def test_excludes_reference_movie() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    other = make_movie("other", "Other", ["Drama"])

    result = recommend_movies(reference, [reference, other])

    assert [item.movie for item in result] == [other]


def test_compares_genres_without_case_sensitivity() -> None:
    reference = make_movie("ref", "Reference", ["sCi-Fi"])
    candidate = make_movie("candidate", "Candidate", ["SCI-FI"])

    result = recommend_movies(reference, [candidate])

    assert result[0].shared_genres_count == 1


def test_breaks_ties_by_title_then_id() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    same_title_later_id = make_movie("z-id", "Alpha", ["Drama"])
    beta = make_movie("b-id", "Beta", ["Drama"])
    same_title_earlier_id = make_movie("a-id", "Alpha", ["Drama"])

    result = recommend_movies(
        reference, [beta, same_title_later_id, same_title_earlier_id]
    )

    assert [item.movie.id for item in result] == ["a-id", "z-id", "b-id"]


def test_applies_limit() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    candidates = [
        make_movie(f"movie-{index}", f"Movie {index}", ["Drama"])
        for index in range(3)
    ]

    result = recommend_movies(reference, candidates, limit=2)

    assert len(result) == 2


def test_empty_catalog_returns_no_recommendations() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])

    assert recommend_movies(reference, []) == []


def test_reference_with_no_genres_returns_no_recommendations() -> None:
    reference = make_movie("ref", "Reference", [])
    candidate = make_movie("candidate", "Candidate", ["Drama"])

    assert recommend_movies(reference, [candidate]) == []


def test_does_not_return_duplicate_movies() -> None:
    reference = make_movie("ref", "Reference", ["Drama"])
    candidate = make_movie("candidate", "Candidate", ["Drama"])

    result = recommend_movies(reference, [candidate, candidate])

    assert [item.movie for item in result] == [candidate]
