import os
from socket import gethostbyname, gethostname
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

from kombu.utils.url import safequote

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY')
DEBUG = os.getenv('DEBUG') != 'off'

allowed_hosts = os.getenv('ALLOWED_HOSTS')

if allowed_hosts:
    hosts = [h.strip() for h in allowed_hosts.split(',') if h.strip()]
    ALLOWED_HOSTS = hosts + [gethostbyname(gethostname())]
    CSRF_TRUSTED_ORIGINS = [f'https://*{h}' for h in hosts]

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'processor',
    'django_celery_results',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'video_processor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'video_processor.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'processor'),
        'PORT': os.getenv('DB_PORT', '5432'),
        'USER': os.getenv('DB_USER', 'processor'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'processor-password'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

STATIC_URL = 'static/'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

storages_config = {}

STORAGE_DIR = os.getenv('STORAGE_MOUNT', os.path.join(os.path.dirname(__file__), '..'))

if os.getenv('CLEANUP_AFTER_PROCESSING') == 'yes':
    CLEANUP_AFTER_PROCESSING = True
else:
    CLEANUP_AFTER_PROCESSING = False

if os.getenv('USE_CPU_ONLY') == 'true':
    USE_CPU_ONLY = True
else:
    USE_CPU_ONLY = False

if os.getenv('USE_SYNCHRONOUS_PROCESSING') == 'true':
    USE_SYNCHRONOUS_PROCESSING = True
else:
    USE_SYNCHRONOUS_PROCESSING = False

# Tracking settings - filter static false positives by tracking detections across frames
if os.getenv('USE_TRACKING') == 'false':
    USE_TRACKING = False
else:
    USE_TRACKING = True  # Enabled by default

# Minimum movement as fraction of frame diagonal to be considered "moving"
# 0.05 = 5% of frame diagonal = ~54 pixels on 1080p (~96 pixels diagonally)
MIN_DISPLACEMENT_THRESHOLD = float(os.getenv('MIN_DISPLACEMENT_THRESHOLD', '0.05'))

# Minimum IoU for detection-track association
TRACKING_IOU_THRESHOLD = float(os.getenv('TRACKING_IOU_THRESHOLD', '0.3'))

# Maximum frames a track can be lost before deletion
TRACKING_MAX_AGE = int(os.getenv('TRACKING_MAX_AGE', '30'))

# High-confidence override for static tracks:
# Tracks shorter than this duration (seconds) with high confidence are kept
# even if they don't move much (e.g., deer standing still munching grass)
TRACKING_MAX_STATIC_DURATION = float(os.getenv('TRACKING_MAX_STATIC_DURATION', '120'))

# Minimum confidence to override static classification for short tracks
# MegaDetector is more confident about real animals than false positives
TRACKING_MIN_CONFIDENCE_OVERRIDE = float(os.getenv('TRACKING_MIN_CONFIDENCE_OVERRIDE', '0.80'))

# Detection threshold - lowered now that tracking filters false positives
# Was 0.65, now 0.4 to catch more potential animals
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.4'))

if os.getenv('AWS_ACCESS_KEY_ID'):
    FRAMES_BUCKET = os.getenv('FRAMES_BUCKET')
    VIDEOS_BUCKET = os.getenv('VIDEOS_BUCKET')
    
    if not FRAMES_BUCKET or not VIDEOS_BUCKET:
        raise ImproperlyConfigured('When configuring for AWS, you need to set FRAMES_BUCKET and VIDEOS_BUCKET as env vars')
    storages_config = { 
        "frames": {
            "BACKEND": "storages.backends.s3.S3Storage",
            "OPTIONS": {
                "bucket_name": FRAMES_BUCKET
            },
        },
        "videos": {
            "BACKEND": "storages.backends.s3.S3Storage",
            "OPTIONS": {
                "bucket_name": VIDEOS_BUCKET
            },
        },
    }
else:
    storages_config = {
        "frames": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {
                "location": f'{STORAGE_DIR}/frames',
                "base_url": "/media/frames/",
            },
        },
        "videos": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {
                "location": f'{STORAGE_DIR}/videos',
                "base_url": "/media/videos/",
            },
        },
    }

STORAGES = {
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
    **storages_config
}

MEDIA_ROOT = STORAGE_DIR
MEDIA_URL = '/media/'

# CELERY settings

REDIS_HOST = os.getenv('REDIS_HOST')

if os.getenv('AWS_ACCESS_KEY_ID'):
    CELERY_BROKER_URL = 'sqs://'
    CELERY_BROKER_TRANSPORT_OPTIONS = {
        'region': os.getenv('AWS_REGION'),
    }
elif REDIS_HOST:
    CELERY_BROKER_URL = f"redis://{REDIS_HOST}:6379/0"
else:
    raise ImproperlyConfigured('Set either a REDIS_HOST or your AWS credentials for SQS as env vars')

CELERY_TASK_ROUTES = {
    'processor.tasks.chunk_video': {
        'queue': 'chunk_video'
    },
    'processor.tasks.extract_frames': {
        'queue': 'extract'
    },
    'processor.tasks.detect': {
        'queue': 'detect'
    },
    'processor.tasks.track_and_filter_detections': {
        'queue': 'create_output'
    },
    'processor.tasks.draw_detections': {
        'queue': 'create_output'
    },
    'processor.tasks.find_completed_videos': {
        'queue': 'create_output'
    },
    'processor.tasks.make_detection_video': {
        'queue': 'create_output'
    },
}
CELERY_RESULT_BACKEND = 'django-db'

# Task acknowledgment settings - critical for surviving worker restarts
# With acks_late=True, tasks are only acknowledged AFTER completion.
# If a worker dies mid-task, the message stays in Redis and gets redelivered.
CELERY_TASK_ACKS_LATE = True

# Reject tasks when worker is shutting down so they get requeued
CELERY_TASK_REJECT_ON_WORKER_LOST = True

CELERY_BEAT_SCHEDULE = {
    'find-completed-videos': {
        'task': 'processor.tasks.find_completed_videos',
        'schedule': 10.0,
    }
}
