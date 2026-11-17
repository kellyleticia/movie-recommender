FROM python:3.13-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY pyproject.toml ./
COPY --chown=app:app \
    app/main.py \
    app/catalog.py \
    app/models.py \
    app/routes.py \
    app/recommender.py \
    app/content_recommender.py \
    app/similarity.py \
    app/text_features.py \
    ./app/

RUN python -m pip install --no-cache-dir .

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
