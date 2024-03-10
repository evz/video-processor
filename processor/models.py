import os
from django.db import models
from django.core.files.storage import storages


STATI = {
    'ENQUEUED': 'Enqueued',
    'PROCESSING': 'Processing',
    'COMPLETED': 'Completed',
    'FAILED': 'Failed',
}

DETECTION_CATEGORIES = {
    '1': 'Animal',
    '2': 'Person',
    '3': 'Vehicle',
}


class Video(models.Model):
    name = models.CharField(max_length=1000)
    video_file = models.FileField(storage=storages['videos'], null=True)
    frame_count = models.IntegerField()
    status = models.CharField(max_length=10, choices=STATI)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name


def frame_upload_path(instance, filename):
    video_name = os.path.splitext(instance.video.name)[0]
    return f'{video_name}/frame{instance.frame_number:07d}.jpg'


class Frame(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    frame_file = models.FileField(storage=storages['frames'], 
                                  null=True, 
                                  upload_to=frame_upload_path)
    frame_number = models.IntegerField()
    status = models.CharField(max_length=10, choices=STATI, db_index=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.video.name} - Frame {self.frame_number}'


class Detection(models.Model):
    frame = models.ForeignKey(Frame, on_delete=models.CASCADE)
    category = models.CharField(choices=DETECTION_CATEGORIES, max_length=1)
    confidence = models.FloatField()
    x_coord = models.FloatField()
    y_coord = models.FloatField()
    box_width = models.FloatField()
    box_height = models.FloatField()

    def __str__(self):
        return f'{self.frame} ({self.x_coord}, {self.y_coord}, {self.box_width}, {self.box_height})'
