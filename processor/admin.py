from django.contrib import admin

from .models import Video


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    fields = ['video_file', 'detections_file']
    readonly_fields = [
        'name',
        'frame_count',
        'status',
        'created',
        'updated',
    ]

    list_display = [
        'name',
        'status',
        'frame_count',
        'formatted_created',
        'formatted_updated',
    ]

    def format_date(self, date):
        return date.strftime('%Y-%m-%d %H:%M:%S')

    def formatted_created(self, obj):
        return self.format_date(obj.created)

    def formatted_updated(self, obj):
        return self.format_date(obj.updated)
