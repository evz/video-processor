#!/bin/bash
set -e

python3 manage.py migrate --noinput

# Try to make a superuser and just skip it on failure
python3 manage.py createsuperuser --noinput || true

exec "$@"
