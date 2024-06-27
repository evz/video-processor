from django.dispatch import receiver
from django.db.models.signals import post_save

from processor.models import Video
from processor.tasks import analyze


@receiver(post_save, sender=Video)
def trigger_chunk_video(sender, **kwargs):
    video = kwargs['instance']
    analyze.delay(video.id)
