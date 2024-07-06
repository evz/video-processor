from django.contrib import admin

from .models import Video, Frame, Detection


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    
    def change_view(self, request, object_id, form_url='', extra_context=None):
        extra_context = extra_context or {}
        
        video_id = object_id.split('/')[0]
        detections_file = Video.objects.get(id=video_id).detections_file

        if detections_file:
            extra_context['detections_video_url'] = detections_file.url
        
        return super().change_view(request, object_id, form_url, extra_context=extra_context)
        

    change_form_template = 'admin/video_change_form.html'
    fields = ['video_file']
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
        'progress',
        'formatted_created',
        'formatted_updated',
        'detection_count',
        'detection_video_available',
    ]

    def format_date(self, date):
        return date.strftime('%Y-%m-%d %H:%M:%S')

    def formatted_created(self, obj):
        return self.format_date(obj.created)

    def formatted_updated(self, obj):
        return self.format_date(obj.updated)
    
    def progress(self, obj):
        completed_frames = Frame.objects.filter(video_chunk__video=obj).filter(status='COMPLETED').count()
        return f'{completed_frames} / {obj.frame_count}'
    
    def detection_video_available(self, obj):
        if obj.detections_file != '':
            return 'Yes'
    
    def detection_count(self, obj):
        return Detection.objects.filter(frame__video_chunk__video=obj).count()

    formatted_created.short_description = 'Created'
    formatted_updated.short_description = 'Updated'
