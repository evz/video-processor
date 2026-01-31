#!/usr/bin/env python3
"""
Autoscale workers based on detect queue depth and disk usage.

Rules:
- If disk usage >= 90%, scale ALL workers to 0 (emergency stop)
- Otherwise: detect queue < 100k -> 2 extract workers, else 1
"""

import os
import shutil
import subprocess
import time

import redis

REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '30'))
QUEUE_THRESHOLD = int(os.getenv('QUEUE_THRESHOLD', '100000'))
EXTRACT_HIGH = int(os.getenv('EXTRACT_HIGH', '2'))
EXTRACT_LOW = int(os.getenv('EXTRACT_LOW', '1'))
COMPOSE_FILE = os.getenv('COMPOSE_FILE', 'docker-compose-local.yaml')
COMPOSE_PROJECT = os.getenv('COMPOSE_PROJECT_NAME', 'video-processor')
COMPOSE_PROJECT_DIR = os.getenv('COMPOSE_PROJECT_DIR', '/workspace')
STORAGE_MOUNT = os.getenv('STORAGE_MOUNT', '/output')
DISK_THRESHOLD = int(os.getenv('DISK_THRESHOLD', '90'))  # percent

# Queue names (must match CELERY_TASK_ROUTES in settings.py)
QUEUES = ['chunk_video', 'extract', 'detect', 'create_output']

# Services to scale down on disk emergency
ALL_SERVICES = ['chunk_video', 'extract', 'detect', 'create_output']


def get_queue_length(redis_client, queue_name):
    """Get the number of messages in a Celery queue."""
    return redis_client.llen(queue_name)


def get_disk_usage_percent(path):
    """Get disk usage percentage for the given path."""
    try:
        usage = shutil.disk_usage(path)
        return (usage.used / usage.total) * 100
    except Exception as e:
        print(f'Error checking disk usage: {e}', flush=True)
        return None


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f'{num_bytes:.1f}{unit}'
        num_bytes /= 1024.0
    return f'{num_bytes:.1f}PB'


def get_running_containers(service_name):
    """Get the number of running containers for a service."""
    cmd = [
        'docker', 'compose',
        '-p', COMPOSE_PROJECT,
        '-f', COMPOSE_FILE,
        '--project-directory', COMPOSE_PROJECT_DIR,
        'ps', '--format', 'json', service_name
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            import json
            lines = result.stdout.strip().split('\n')
            running = sum(1 for line in lines if line.strip() and json.loads(line).get('State') == 'running')
            return running
    except Exception:
        pass
    return None


def scale_service(service_name, replicas):
    """Scale a service using docker compose."""
    cmd = [
        'docker', 'compose',
        '-p', COMPOSE_PROJECT,
        '-f', COMPOSE_FILE,
        '--project-directory', COMPOSE_PROJECT_DIR,
        'up', '-d', '--scale', f'{service_name}={replicas}', '--no-recreate', service_name
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f'  Error scaling {service_name}: {result.stderr}', flush=True)
            return False
        return True
    except Exception as e:
        print(f'  Error scaling {service_name}: {e}', flush=True)
        return False


def main():
    print('Autoscaler starting', flush=True)
    print(f'  Redis: {REDIS_HOST}', flush=True)
    print(f'  Queue threshold: {QUEUE_THRESHOLD}', flush=True)
    print(f'  Disk threshold: {DISK_THRESHOLD}%', flush=True)
    print(f'  Extract high/low: {EXTRACT_HIGH}/{EXTRACT_LOW}', flush=True)
    print(f'  Check interval: {CHECK_INTERVAL}s', flush=True)
    print(f'  Storage path: {STORAGE_MOUNT}', flush=True)
    print(f'  Project dir: {COMPOSE_PROJECT_DIR}', flush=True)

    redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    # Wait for Redis
    while True:
        try:
            redis_client.ping()
            print('Connected to Redis', flush=True)
            break
        except redis.ConnectionError:
            print('Waiting for Redis...', flush=True)
            time.sleep(5)

    current_extract_scale = None
    emergency_mode = False

    while True:
        try:
            timestamp = time.strftime('%H:%M:%S')

            # Get all queue lengths
            queue_lengths = {q: get_queue_length(redis_client, q) for q in QUEUES}
            queue_str = ', '.join(f'{q}={length}' for q, length in queue_lengths.items())

            # Get disk usage
            disk_percent = get_disk_usage_percent(STORAGE_MOUNT)
            if disk_percent is not None:
                usage = shutil.disk_usage(STORAGE_MOUNT)
                disk_str = f'{disk_percent:.1f}% ({format_bytes(usage.free)} free)'
            else:
                disk_str = 'unknown'

            # Get current extract worker count
            extract_running = get_running_containers('extract')
            extract_str = str(extract_running) if extract_running is not None else '?'

            print(f'[{timestamp}] queues: {queue_str} | disk: {disk_str} | extract workers: {extract_str}', flush=True)

            # Check for disk emergency
            if disk_percent is not None and disk_percent >= DISK_THRESHOLD:
                if not emergency_mode:
                    print(f'  DISK EMERGENCY: {disk_percent:.1f}% >= {DISK_THRESHOLD}%, scaling all workers to 0', flush=True)
                    for service in ALL_SERVICES:
                        scale_service(service, 0)
                    emergency_mode = True
                    current_extract_scale = 0
            else:
                # Normal operation
                if emergency_mode:
                    print(f'  Disk usage back to normal, resuming operations', flush=True)
                    emergency_mode = False

                # Apply extract scaling rule
                detect_queue_len = queue_lengths['detect']
                if detect_queue_len < QUEUE_THRESHOLD:
                    desired_extract_scale = EXTRACT_HIGH
                else:
                    desired_extract_scale = EXTRACT_LOW

                if desired_extract_scale != current_extract_scale:
                    print(f'  Scaling extract: {current_extract_scale} -> {desired_extract_scale}', flush=True)
                    if scale_service('extract', desired_extract_scale):
                        current_extract_scale = desired_extract_scale

        except redis.ConnectionError as e:
            print(f'Redis connection error: {e}', flush=True)
        except Exception as e:
            print(f'Error: {e}', flush=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == '__main__':
    main()
