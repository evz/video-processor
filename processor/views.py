import re
from datetime import datetime

from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from django.db.models import Count
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils.text import slugify

from .models import Video, Detection, Tag

# Minimum number of detections for a video to appear in the list
MIN_DETECTIONS = 10

# Regex pattern for ch<channel>_<YYYYMMDD><HHMMSS> video names
VIDEO_NAME_PATTERN = re.compile(r'^ch(\d+)_(\d{8})(\d{6})$')


def parse_video_datetime(name):
    """Parse datetime from video name like ch02_20230712115149."""
    if not name:
        return None
    match = VIDEO_NAME_PATTERN.match(name)
    if match:
        channel = match.group(1)
        date_str = match.group(2)
        time_str = match.group(3)
        try:
            dt = datetime.strptime(f'{date_str}{time_str}', '%Y%m%d%H%M%S')
            return {'datetime': dt, 'channel': int(channel)}
        except ValueError:
            pass
    return None


def video_list(request):
    """List all completed videos with detection files and enough detections."""
    videos = Video.objects.filter(
        status='COMPLETED',
    ).exclude(
        detections_file=''
    ).annotate(
        detection_count=Count('videochunk__frame__detection')
    ).filter(
        detection_count__gte=MIN_DETECTIONS
    )

    # Filter by tag if specified
    tag_filter = request.GET.get('tag')
    active_tag = None
    if tag_filter:
        active_tag = Tag.objects.filter(slug=tag_filter).first()
        if active_tag:
            videos = videos.filter(tags=active_tag)

    # Filter by channel if specified
    channel_filter = request.GET.get('channel')
    active_channel = None
    if channel_filter:
        active_channel = channel_filter
        videos = videos.filter(name__startswith=f'ch{channel_filter.zfill(2)}_')

    # Get available channels for filter sidebar
    all_video_names = Video.objects.filter(
        status='COMPLETED'
    ).exclude(
        detections_file=''
    ).values_list('name', flat=True)

    channels_set = set()
    for name in all_video_names:
        parsed = parse_video_datetime(name)
        if parsed:
            channels_set.add(parsed['channel'])
    available_channels = sorted(channels_set)

    # Get all tags with video counts for the filter sidebar
    all_tags = Tag.objects.all()
    tags_with_counts = []
    for tag in all_tags:
        tag_videos = tag.videos.filter(
            status='COMPLETED'
        ).exclude(
            detections_file=''
        ).annotate(
            detection_count=Count('videochunk__frame__detection')
        ).filter(
            detection_count__gte=MIN_DETECTIONS
        )
        if channel_filter:
            tag_videos = tag_videos.filter(name__startswith=f'ch{channel_filter.zfill(2)}_')
        count = tag_videos.count()
        if count > 0:
            tags_with_counts.append({'tag': tag, 'count': count})

    # Parse datetime for each video and build video_data list
    video_data = []
    for video in videos:
        parsed = parse_video_datetime(video.name)
        video_data.append({
            'video': video,
            'captured_at': parsed['datetime'] if parsed else None,
            'channel': parsed['channel'] if parsed else None,
            'tags': video.tags.all(),
        })

    # Sort by captured_at (oldest to newest), with None values at the end
    video_data.sort(key=lambda x: (x['captured_at'] is None, x['captured_at']))

    # Paginate
    paginator = Paginator(video_data, 24)  # 24 videos per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'processor/video_list.html', {
        'page_obj': page_obj,
        'total_count': len(video_data),
        'tags_with_counts': tags_with_counts,
        'active_tag': active_tag,
        'available_channels': available_channels,
        'active_channel': active_channel,
    })


def video_detail(request, video_id):
    """Show a single video with its detections."""
    video = get_object_or_404(Video, id=video_id)
    parsed = parse_video_datetime(video.name)

    # Get total detections
    total_detections = Detection.objects.filter(
        frame__video_chunk__video=video
    ).count()

    # Get all available tags for the tag selector
    all_tags = Tag.objects.all()
    video_tags = video.tags.all()

    return render(request, 'processor/video_detail.html', {
        'video': video,
        'captured_at': parsed['datetime'] if parsed else None,
        'channel': parsed['channel'] if parsed else None,
        'total_detections': total_detections,
        'all_tags': all_tags,
        'video_tags': video_tags,
    })


@require_POST
def add_tag(request, video_id):
    """Add a tag to a video."""
    video = get_object_or_404(Video, id=video_id)
    tag_name = request.POST.get('tag_name', '').strip()
    tag_id = request.POST.get('tag_id')

    if tag_id:
        # Adding existing tag
        tag = get_object_or_404(Tag, id=tag_id)
    elif tag_name:
        # Creating new tag
        slug = slugify(tag_name)
        tag, created = Tag.objects.get_or_create(
            slug=slug,
            defaults={'name': tag_name}
        )
    else:
        return redirect('video_detail', video_id=video_id)

    video.tags.add(tag)
    return redirect('video_detail', video_id=video_id)


@require_POST
def remove_tag(request, video_id, tag_id):
    """Remove a tag from a video."""
    video = get_object_or_404(Video, id=video_id)
    tag = get_object_or_404(Tag, id=tag_id)
    video.tags.remove(tag)
    return redirect('video_detail', video_id=video_id)
