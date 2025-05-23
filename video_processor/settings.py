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
    ALLOWED_HOSTS = [allowed_hosts, gethostbyname(gethostname())]
    CSRF_TRUSTED_ORIGINS = [f'https://*{allowed_hosts}']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
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
                "base_url": "frames",
            },
        },
        "videos": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
            "OPTIONS": {
                "location": f'{STORAGE_DIR}/videos',
                "base_url": "videos",
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

CELERY_BEAT_SCHEDULE = {
    'find-completed-videos': {
        'task': 'processor.tasks.find_completed_videos',
        'schedule': 10.0,
    }
}
