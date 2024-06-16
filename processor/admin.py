from django.contrib import admin

from .models import Video


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    fields = ['video_file']
    readonly_fields = [
        'name',
        'frame_count',
        'status',
        'created',
        'updated',
    ]
