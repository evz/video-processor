#!/bin/bash
set -e

python manage.py migrate --noinput

# Try to make a superuser and just skip it on failure
python manage.py createsuperuser --noinput || true

exec "$@"
