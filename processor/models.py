import os
from typing import Dict, Tuple, Literal
from django.db import models
from django.core.files.storage import storages


STATI: Dict[str, str] = {
    'ENQUEUED': 'Enqueued',
    'PROCESSING': 'Processing',
    'COMPLETED': 'Completed',
    'FAILED': 'Failed',
}

DETECTION_CATEGORIES: Dict[str, str] = {
    '1': 'Animal',
    '2': 'Person',
    '3': 'Vehicle',
}

StatusChoice = Literal['ENQUEUED', 'PROCESSING', 'COMPLETED', 'FAILED']
CategoryChoice = Literal['1', '2', '3']


class Tag(models.Model):
    """Tags for categorizing videos by animal type or other criteria."""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            from django.utils.text import slugify
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)


class Video(models.Model):
    name = models.CharField(max_length=1000, null=True)
    video_file = models.FileField(storage=storages['videos'])
    detections_file = models.FileField(storage=storages['videos'], null=True)
    frame_count = models.IntegerField(null=True)
    frame_rate = models.FloatField(default=20.0)
    status = models.CharField(max_length=10, choices=STATI, default='ENQUEUED')
    tags = models.ManyToManyField(Tag, blank=True, related_name='videos')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'Video {self.id}'


class VideoChunk(models.Model):
    name = models.CharField(max_length=500)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    video_file = models.FileField(storage=storages['videos'])
    created = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATI, default='ENQUEUED')
    sequence_number = models.IntegerField(default=1)
    frame_count = models.IntegerField(null=True, help_text="Number of frames in this chunk")
    start_frame = models.IntegerField(null=True, help_text="First frame number in this chunk")
    end_frame = models.IntegerField(null=True, help_text="Last frame number in this chunk")

    def __str__(self):
        return f'VideoChunk - {self.name} from Video {self.video.id}'


def frame_upload_path(instance: 'Frame', filename: str) -> str:
    """Generate upload path for frame images."""
    video_name = os.path.splitext(instance.video_chunk.video_file.name)[0]
    return f'{video_name}/frame{instance.frame_number:07d}.jpg'


def detection_upload_path(instance: 'Frame', filename: str) -> str:
    """Generate upload path for detection result images."""
    video_name = os.path.splitext(instance.video_chunk.video_file.name)[0]
    return f'{video_name}/detections/frame{instance.frame_number:07d}.jpg'


class Frame(models.Model):
    video_chunk = models.ForeignKey(VideoChunk, null=True, on_delete=models.CASCADE)
    frame_file = models.FileField(storage=storages['frames'], 
                                  null=True, 
                                  upload_to=frame_upload_path)
    detections_file = models.FileField(storage=storages['frames'], 
                                       null=True, 
                                       upload_to=detection_upload_path)
    frame_number = models.IntegerField()
    status = models.CharField(max_length=10, choices=STATI, db_index=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.video_chunk.name} - Frame {self.frame_number}'


class Track(models.Model):
    """Represents a tracked object across multiple frames."""
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    track_id = models.IntegerField()  # Unique ID within video
    category = models.CharField(choices=DETECTION_CATEGORIES, max_length=1)
    start_frame = models.IntegerField()
    end_frame = models.IntegerField()
    total_displacement = models.FloatField(default=0.0)  # Movement metric (fraction of frame diagonal)
    is_static = models.BooleanField(default=False)  # Flagged as non-moving (false positive)

    class Meta:
        unique_together = ['video', 'track_id']

    def __str__(self):
        return f'Track {self.track_id} for Video {self.video.id} ({self.start_frame}-{self.end_frame})'


class Detection(models.Model):
    frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
    category = models.CharField(choices=DETECTION_CATEGORIES, max_length=1)
    confidence = models.FloatField()
    x_coord = models.FloatField()
    y_coord = models.FloatField()
    box_width = models.FloatField()
    box_height = models.FloatField()
    track = models.ForeignKey(Track, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f'{self.frame} ({self.x_coord}, {self.y_coord}, {self.box_width}, {self.box_height})'
