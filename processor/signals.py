from django.dispatch import receiver
from django.db.models.signals import post_save

from processor.models import Video
from processor.tasks import analyze, extract_frames

# Commenting this for now since batching the frame extraction seems far slower
# than just having one process work on it
# @receiver(post_save, sender=Video)
def trigger_analyze(sender, **kwargs):
    video = kwargs['instance']
    analyze.delay(video.id)


@receiver(post_save, sender=Video)
def trigger_extract_frame(sender, **kwargs):
    video = kwargs['instance']
    extract_frames.delay(video.id)
