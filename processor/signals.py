from django.dispatch import receiver
from django.db.models.signals import post_save

from processor.models import Video
from processor.tasks import extract_frames


@receiver(post_save, sender=Video)
def trigger_extract(sender, **kwargs):
    video = kwargs['instance']
    extract_frames.delay(video.id)
