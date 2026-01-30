from django.contrib import admin

from .models import Video, Frame, Detection, Tag


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'video_count', 'created']
    search_fields = ['name']
    prepopulated_fields = {'slug': ('name',)}

    def video_count(self, obj):
        return obj.videos.count()
    video_count.short_description = 'Videos'


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
    fields = ['video_file', 'tags']
    readonly_fields = [
        'name',
        'frame_count',
        'status',
        'created',
        'updated',
    ]
    filter_horizontal = ['tags']

    list_display = [
        'name',
        'status',
        'progress',
        'formatted_created',
        'formatted_updated',
        'detection_count',
        'detection_video_available',
    ]

    list_filter = ['status']

    def format_date(self, date):
        return date.strftime('%Y-%m-%d %H:%M:%S')

    def formatted_created(self, obj):
        return self.format_date(obj.created)

    def formatted_updated(self, obj):
        return self.format_date(obj.updated)
    
    def progress(self, obj):
        extracted_frames = Frame.objects.filter(video_chunk__video=obj).count()
        completed_frames = Frame.objects.filter(video_chunk__video=obj, status='COMPLETED').count()
        return f'{completed_frames} / {extracted_frames} / {obj.frame_count}'
    
    def detection_video_available(self, obj):
        if obj.detections_file != '':
            return 'Yes'
    
    def detection_count(self, obj):
        return Detection.objects.filter(frame__video_chunk__video=obj).count()

    formatted_created.short_description = 'Created'
    formatted_updated.short_description = 'Updated'
    progress.short_description = 'Processed / Extracted / Total'
