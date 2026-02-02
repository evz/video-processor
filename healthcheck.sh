#!/bin/bash
# Healthcheck script for Celery workers
# Verifies the heartbeat file was modified within the last N minutes

HEARTBEAT_FILE="/tmp/celery_heartbeat"
MAX_AGE_MINUTES="${HEALTHCHECK_MAX_AGE:-5}"

# If heartbeat file doesn't exist, worker hasn't started processing yet
# Give it a grace period (handled by Docker's start_period)
if [ ! -f "$HEARTBEAT_FILE" ]; then
    exit 0
fi

# Check if file was modified within the threshold
if find "$HEARTBEAT_FILE" -mmin -"$MAX_AGE_MINUTES" | grep -q .; then
    exit 0
else
    echo "Heartbeat file is stale (older than $MAX_AGE_MINUTES minutes)"
    exit 1
fi
