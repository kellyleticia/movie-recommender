# Movie Recommender API

A small REST API that recommends movies based on genres shared with a reference
movie. The recommender uses a transparent rule-based ranking algorithm; it does
not use trained machine learning models.

## Features

- An in-memory movie catalog with lookup by movie ID.
- Case-insensitive genre comparison.
- Recommendations ranked by the number of distinct shared genres.
- Alphabetical tie-breaking by title, then movie ID.
- Configurable recommendation limits through the API.

## Tech stack

- Python 3.13
- FastAPI
- Pydantic
- Pytest
- Ruff for linting

## Project structure

```text
.
├── app/
│   ├── catalog.py       # In-memory movie catalog
│   ├── main.py          # FastAPI application factory and Uvicorn entry point
│   ├── models.py        # Pydantic Movie model
│   ├── recommender.py   # Rule-based recommendation algorithm
│   └── routes.py        # HTTP endpoint definitions
├── tests/
│   ├── test_api.py
│   ├── test_catalog.py
│   ├── test_models.py
│   └── test_recommender.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Requirements and installation

Python 3.13 is required (`pyproject.toml` specifies `>=3.13,<3.14`). From the
repository root, create and activate a virtual environment, then install the
project and its development dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Running the API

From the repository root, start the development server with:

```bash
uvicorn app.main:app --reload
```

The application is also exposed as `app` from `app.main`, and `create_app`
accepts an optional `MovieCatalog` for configuring an application instance.

## API endpoints

| Method and path | Behavior | Responses |
| --- | --- | --- |
| `GET /health` | Health check. | `200`: `{"status":"ok"}` |
| `GET /movies` | Lists movies in catalog insertion order. Each movie has `id`, `title`, and `genres`. | `200`: a JSON array; an empty catalog returns `[]`. |
| `GET /movies/{movie_id}/recommendations` | Recommends movies for the given ID. Accepts the optional `limit` query parameter (default `10`, minimum `1`, maximum `50`). | `200`: `movie_id` and a `recommendations` array; `404` if the reference movie is absent; `422` if `limit` is outside the allowed range. |

Each recommendation contains `id`, `title`, `genres`, and
`shared_genres_count`. A valid reference movie with no matches returns `200`
with an empty recommendations array.

## Example requests

The default demonstration catalog contains *Arrival* (`arrival`) and *Contact*
(`contact`) with Drama and Sci-Fi genres, and *Amélie* (`amelie`) with Comedy
and Romance genres.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

List the demonstration movies:

```bash
curl http://127.0.0.1:8000/movies
```

Get up to five recommendations for *Arrival*:

```bash
curl 'http://127.0.0.1:8000/movies/arrival/recommendations?limit=5'
```

The recommendation response includes *Contact* with `shared_genres_count` of
`2`; *Arrival* is excluded, and *Amélie* has no shared genres.

## Recommendation logic

The algorithm converts each movie's genres to the normalized, case-folded set
provided by the `Movie` model. It intersects the reference movie's set with
each catalog movie's set, excludes the reference and repeated movie IDs, and
keeps candidates with at least one shared genre. Results are sorted by shared
genre count descending, followed by case-insensitive title and ID order. The
requested limit is applied to the sorted results.

## Testing and linting

Run the test suite and Ruff from the repository root:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

## Limitations

The catalog is held in memory, so changes are not persisted across application
restarts. When no catalog is passed to `create_app`, the API uses the three
demonstration movies described above. Recommendations use genre overlap rules;
there is no machine learning model or persistent data store.
