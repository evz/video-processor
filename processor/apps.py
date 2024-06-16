from django.apps import AppConfig
from django.db.models.signals import post_save


class ProcessorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'processor'
    
    def ready(self):
        from . import signals
