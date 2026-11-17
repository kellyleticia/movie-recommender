#!/usr/bin/env bash

set -euo pipefail

BASE_URL="http://127.0.0.1:8000"

printf '=== API health ===\n'
curl --fail --silent --show-error "$BASE_URL/health"
printf '\n\n'

printf '=== Movie catalog ===\n'
curl --fail --silent --show-error "$BASE_URL/movies"
printf '\n\n'

printf '=== Genre-based recommendations for Arrival ===\n'
curl --fail --silent --show-error \
    "$BASE_URL/movies/arrival/recommendations"
printf '\n\n'

printf '=== Content-based recommendations for Arrival ===\n'
curl --fail --silent --show-error \
    "$BASE_URL/movies/arrival/content-recommendations"
printf '\n'
