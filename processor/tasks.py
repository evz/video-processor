import os
import subprocess
import logging
import math
import json
import time
import itertools
import shutil
import glob
import sys

from io import BytesIO
from pathlib import Path
from threading import Thread, Event
from typing import List, Dict, Optional, Union, TypedDict, Tuple

import numpy
from PIL import Image
import cv2
from megadetector.visualization.visualization_utils import render_detection_bounding_boxes
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from django.conf import settings
from django.core.files import File
from django.db.models import Sum, Count, Q
from celery import chain
from celery.signals import worker_process_init, task_failure, task_postrun
from video_processor import celery_app

from .models import Video, VideoChunk, Frame, Detection, Track

logger = logging.getLogger(__name__)

# GPU memory threshold - exit worker if usage exceeds this percentage
GPU_MEMORY_THRESHOLD = 0.90  # 90%


def get_gpu_memory_usage():
    """
    Get current GPU memory usage as a fraction (0.0 to 1.0).
    Returns None if unable to query GPU.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            line = result.stdout.strip().split('\n')[0]
            used, total = map(int, line.split(','))
            return used / total if total > 0 else None
    except Exception as e:
        logger.warning(f'Failed to query GPU memory: {e}')
    return None


def check_gpu_memory_and_exit_if_high():
    """
    Check GPU memory usage and exit the worker if it's too high.
    Docker will restart the container, clearing GPU memory.
    """
    usage = get_gpu_memory_usage()
    if usage is not None and usage > GPU_MEMORY_THRESHOLD:
        logger.warning(
            f'GPU memory usage at {usage:.1%}, exceeds threshold of {GPU_MEMORY_THRESHOLD:.0%}. '
            f'Exiting worker to free memory.'
        )
        sys.exit(1)


class DetectionResult(TypedDict):
    """Type definition for MegaDetector detection results."""
    category: str
    conf: float
    bbox: List[float]


def get_video_metadata_ffprobe(filepath: str) -> Tuple[float, int]:
    """
    Get video metadata using ffprobe (fast, reads header only).

    Returns:
        Tuple of (frame_rate, estimated_frame_count)
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,duration',
        '-of', 'json',
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe failed: {result.stderr}')

    data = json.loads(result.stdout)
    stream = data.get('streams', [{}])[0]

    # Parse frame rate (format: "30/1" or "30000/1001")
    fps_str = stream.get('avg_frame_rate', '30/1')
    if '/' in fps_str:
        num, den = fps_str.split('/')
        frame_rate = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        frame_rate = float(fps_str)

    # Estimate frame count from duration
    duration = float(stream.get('duration', 0))
    estimated_frame_count = int(duration * frame_rate)

    return frame_rate, estimated_frame_count


class VideoChunkCompleted(FileSystemEventHandler):
    @staticmethod
    def on_closed(event):
        from decord import VideoReader, cpu, gpu
        import gc

        filename = os.path.basename(event.src_path)
        sequence_number = int(filename.split('.')[0][-3:])
        video_name = filename.split(".")[0][:-3]

        video = Video.objects.get(name=video_name)

        # Probe the chunk file to get its frame count
        ctx = gpu(0) if not settings.USE_CPU_ONLY else cpu(0)
        vr = VideoReader(event.src_path, ctx=ctx)
        frame_count = len(vr)

        # Explicitly release GPU memory to prevent CUDA memory leaks
        del vr
        gc.collect()

        with open(event.src_path, 'rb') as f:
            video_chunk = VideoChunk(name=filename,
                                     video=video,
                                     video_file=File(f, filename),
                                     sequence_number=sequence_number,
                                     frame_count=frame_count)

            video_chunk.save()

        logger.info(f'Created chunk {sequence_number} for video {video.id} with {frame_count} frames')

        extract_frames.delay(video_chunk.id)

        os.remove(event.src_path)


class MonitorThread(Thread):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = Observer()
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.is_set()

    def run(self):
        
        output_path = self._args[0]
        
        event_handler = VideoChunkCompleted()
        self.observer.schedule(event_handler, output_path, recursive=True)
        self.observer.start()

        while True:
            if self.stopped():
                break
        
        self.observer.stop()
        self.observer.join()


@celery_app.task(bind=True, max_retries=None)
def chunk_video(self, video_id: int) -> None:
    """
    Break a video into smaller chunks for parallel processing.

    Args:
        video_id: The ID of the Video model instance to process
    """
    # It looks like there's a chance that the message gets picked up by a
    # worker before the row gets saved to the DB, I guess. This is somewhat
    # maddening since this gets triggered by the `post_save` hook but,
    # whatever. Now you know why this retry crap is here.
    retries = 0

    while retries <= 5:
        try:
            video = Video.objects.get(id=video_id)
            break
        except Video.DoesNotExist as e:
            logger.info(f'video {video_id} is not here I guess')
            if retries == 5:
                raise e
            retries += 1
            time.sleep(0.1)

    # Idempotency check: skip if already has chunks (task was redelivered after partial completion)
    existing_chunks = VideoChunk.objects.filter(video=video).count()
    if existing_chunks > 0:
        logger.info(f'Video {video_id} already has {existing_chunks} chunks, skipping chunk_video')
        return

    input_video_dir = os.path.join(settings.STORAGE_DIR, 'videos')
    output_video_dir = os.path.join(settings.STORAGE_DIR, 'output_videos')
    video_basename = os.path.basename(video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_video_dir, video_basename)
    
    os.makedirs(input_video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.join(input_video_dir, video.video_file.name)

    # Get video metadata using ffprobe (fast, header-only read)
    # Frame count is estimated from duration * fps; will be updated with
    # accurate count from chunks after processing
    frame_rate, estimated_frame_count = get_video_metadata_ffprobe(input_filename)

    video.status = 'PROCESSING'
    video.name = video_basename
    video.frame_count = estimated_frame_count
    video.frame_rate = frame_rate
    video.save()

    if not settings.USE_SYNCHRONOUS_PROCESSING:
        monitor_thread = MonitorThread(args=[output_dir])
        monitor_thread.start()
    
    if settings.USE_CPU_ONLY:
        # CPU-only FFmpeg command
        # Force keyframes at segment boundaries to ensure clean cuts without frame overlap
        # Use -fps_mode passthrough to preserve original frame timing (avoids duplicating
        # frames in variable frame rate videos)
        ffmpeg_cmd = [
            'ffmpeg',
            '-i',
            f'{input_filename}',
            '-vf',
            'scale=w=1920:h=1080',
            '-c:v',
            'libx264',
            '-fps_mode',
            'passthrough',
            '-force_key_frames',
            'expr:gte(t,n_forced*30)',
            '-an',
            '-f',
            'segment',
            '-segment_time',
            '00:00:30',
            '-reset_timestamps',
            '1',
            '-map',
            '0',
            f'{output_dir}/{video_basename}%03d.mp4'
        ]
    else:
        # GPU-accelerated FFmpeg command
        # Use NVDEC for hardware decoding (-hwaccel cuda) and NVENC for encoding
        # NVDEC/NVENC are dedicated hardware units, separate from CUDA cores
        # Keep scaling on CPU to avoid competing with detect tasks for CUDA cores
        # Force keyframes at segment boundaries to ensure clean cuts without frame overlap
        # Use -fps_mode passthrough to preserve original frame timing (avoids duplicating
        # frames in variable frame rate videos)
        ffmpeg_cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',          # Use NVDEC for decoding
            '-i',
            f'{input_filename}',
            '-vf',
            'scale=w=1920:h=1080',        # CPU scaling to avoid CUDA core contention
            '-c:v',
            'h264_nvenc',
            '-preset', 'p1',              # Fastest encoding preset
            '-fps_mode',
            'passthrough',
            '-force_key_frames',
            'expr:gte(t,n_forced*30)',
            '-an',
            '-f',
            'segment',
            '-segment_time',
            '00:00:30',
            '-reset_timestamps',
            '1',
            '-map',
            '0',
            f'{output_dir}/{video_basename}%03d.mp4'
        ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    if settings.USE_SYNCHRONOUS_PROCESSING:
        # Process chunks synchronously - find all created chunk files
        chunk_files = sorted(glob.glob(f'{output_dir}/{video_basename}*.mp4'))
        for i, chunk_file in enumerate(chunk_files):
            # Create VideoChunk record
            filename = os.path.basename(chunk_file)
            sequence_number = i  # Use index as sequence number

            # Probe the chunk file to get its frame count
            chunk_vr = VideoReader(chunk_file, ctx=ctx)
            chunk_frame_count = len(chunk_vr)

            with open(chunk_file, 'rb') as f:
                video_chunk = VideoChunk(name=filename,
                                       video=video,
                                       video_file=File(f, filename),
                                       sequence_number=sequence_number,
                                       frame_count=chunk_frame_count)
                video_chunk.save()

            # Process frames synchronously
            extract_frames(video_chunk.id)

            # Clean up chunk file
            os.remove(chunk_file)

        # Check if video is complete and trigger final processing
        find_completed_videos()
    else:
        monitor_thread.stop()
        monitor_thread.join()
    
    os.remove(input_filename)


@celery_app.task(bind=True, max_retries=None)
def extract_frames(self, video_chunk_id: int) -> None:
    """
    Extract individual frames from a video chunk and queue them for detection.

    Args:
        video_chunk_id: The ID of the VideoChunk model instance to process
    """

    # It looks like there's a chance that the message gets picked up by a
    # worker before the row gets saved to the DB, I guess. This is somewhat
    # maddening since this gets triggered by the `post_save` hook but,
    # whatever. Now you know why this retry crap is here.
    retries = 0

    while retries <= 5:
        try:
            video_chunk = VideoChunk.objects.get(id=video_chunk_id)
            break
        except VideoChunk.DoesNotExist as e:
            logger.info(f'video {video_chunk_id} is not here I guess')
            if retries == 5:
                raise e
            retries += 1
            time.sleep(0.1)

    logger.info(f'found video {video_chunk_id}')

    # Calculate start_frame if not already set.
    # This must happen before the status check so retries work correctly.
    if video_chunk.start_frame is None:
        if video_chunk.sequence_number > 0:
            previous_chunk = VideoChunk.objects.filter(
                video=video_chunk.video,
                sequence_number=video_chunk.sequence_number - 1,
                end_frame__isnull=False
            ).first()

            if not previous_chunk:
                # Previous chunk hasn't finished extraction yet, retry later
                logger.info(f'Chunk {video_chunk.sequence_number} waiting for previous chunk to complete')
                raise self.retry(countdown=2)

            start_frame_number = previous_chunk.end_frame + 1
        else:
            start_frame_number = 1

        video_chunk.start_frame = start_frame_number
        video_chunk.end_frame = start_frame_number + video_chunk.frame_count - 1
        video_chunk.save()

    start_frame_number = video_chunk.start_frame

    if video_chunk.status == 'COMPLETED':
        return

    if video_chunk.status == 'PROCESSING':
        # Resuming an interrupted extraction - find already extracted frames
        existing_frame_numbers = set(
            Frame.objects.filter(video_chunk=video_chunk)
            .values_list('frame_number', flat=True)
        )
        logger.info(
            f'Resuming chunk {video_chunk_id}: '
            f'{len(existing_frame_numbers)}/{video_chunk.frame_count} frames already extracted'
        )
    else:
        existing_frame_numbers = set()
        video_chunk.status = 'PROCESSING'
        video_chunk.save()

    output_frames_dir = os.path.join(settings.STORAGE_DIR, 'frames')
    video_basename = os.path.basename(video_chunk.video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_frames_dir, video_basename)

    os.makedirs(output_dir, exist_ok=True)

    # Use decord to extract frames instead of ffmpeg
    # Determine device context (GPU or CPU)
    from decord import VideoReader, cpu, gpu
    import gc

    ctx = gpu(0) if not settings.USE_CPU_ONLY else cpu(0)

    # Open video with decord
    vr = VideoReader(video_chunk.video_file.path, ctx=ctx)

    # Extract and save each frame
    frame_number = int(start_frame_number)
    skipped_frames = 0
    try:
        for frame_idx in range(len(vr)):
            # Skip frames already extracted from a previous interrupted attempt
            if frame_number in existing_frame_numbers:
                frame_number += 1
                continue

            # Get frame as numpy array (RGB format)
            frame_array = vr[frame_idx].asnumpy()

            # Handle empty frames (decord returns empty array when frame can't be decoded,
            # e.g., B-frames referencing data outside this video segment)
            # Check ndim because 0-d arrays have size=1 but shape=()
            if frame_array.ndim < 2 or frame_array.size == 0:
                logger.warning(f'Chunk {video_chunk_id} frame {frame_number}: empty frame from decord, skipping')
                skipped_frames += 1
                frame_number += 1
                continue

            # Convert to PIL Image
            frame_image = Image.fromarray(frame_array)

            # Save frame to BytesIO buffer
            frame_buffer = BytesIO()
            frame_image.save(frame_buffer, format='JPEG', quality=95)
            frame_buffer.seek(0)

            # Create Frame database record
            frame = Frame(video_chunk=video_chunk,
                          frame_file=File(frame_buffer, f'frame{frame_number:07d}.jpg'),
                          detections_file=None,
                          frame_number=frame_number,
                          status='ENQUEUED')
            frame.save()

            if settings.USE_SYNCHRONOUS_PROCESSING:
                detect(frame.id)
            else:
                detect.delay(frame.id)
            frame_number += 1

        if skipped_frames > 0:
            logger.info(f'Chunk {video_chunk_id}: skipped {skipped_frames} undecodable frames')
            # Adjust frame_count to reflect actual extractable frames
            video_chunk.frame_count -= skipped_frames
            video_chunk.end_frame -= skipped_frames
    finally:
        # Explicitly release GPU memory to prevent CUDA memory leaks
        del vr
        gc.collect()

    video_chunk.status = 'COMPLETED'
    video_chunk.save()

    if settings.CLEANUP_AFTER_PROCESSING:
        os.remove(video_chunk.video_file.path)


detector = None


@worker_process_init.connect
def load_model_on_startup(**kwargs):
    """Pre-load the detection model when the worker starts, but only for detect workers."""
    # Only load the model if this worker is handling the detect queue
    # Check command line args for the queue name
    if '-Q' in sys.argv:
        queue_index = sys.argv.index('-Q') + 1
        if queue_index < len(sys.argv) and sys.argv[queue_index] == 'detect':
            global detector
            if detector is None:
                from megadetector.detection.run_detector import load_detector
                logger.info('Loading MegaDetector model at worker startup...')
                detector = load_detector('MDV5A')
                logger.info('MegaDetector model loaded.')


@task_failure.connect
def handle_task_failure(task_id, exception, traceback, **kwargs):
    """
    Handle task failures and exit the worker if a GPU error is detected.
    This allows Docker to restart the container and recover from GPU issues.
    """
    error_msg = str(exception).lower()
    gpu_error_keywords = ['cuda', 'gpu', 'nvenc', 'nvidia', 'device']

    if any(keyword in error_msg for keyword in gpu_error_keywords):
        logger.error(f'GPU error detected in task {task_id}, shutting down worker: {exception}')
        sys.exit(1)


@task_postrun.connect
def check_gpu_memory_after_task(task_id, task, **kwargs):
    """
    Check GPU memory after extract and chunk_video tasks complete.
    If usage exceeds threshold, exit worker so Docker can restart it.
    Skip detect tasks to avoid impacting throughput.
    """
    task_name = task.name if task else ''
    if task_name in ('processor.tasks.extract_frames', 'processor.tasks.chunk_video'):
        check_gpu_memory_and_exit_if_high()


@celery_app.task
def detect(frame_id: int) -> None:
    """
    Run AI detection on a single frame to find animals, people, or vehicles.
    
    Uses MegaDetector v5a model to analyze the frame and store detection results.
    
    Args:
        frame_id: The ID of the Frame model instance to analyze
    """
    
    frame = Frame.objects.get(id=frame_id)

    if frame.status == 'PROCESSING':
        # Resuming after interrupted attempt - clean up partial detections
        Detection.objects.filter(frame=frame).delete()
    elif frame.status != 'ENQUEUED':
        return

    global detector

    if not detector:
        from megadetector.detection.run_detector import load_detector
        detector = load_detector('MDV5A')

    frame.status = 'PROCESSING'
    frame.save()
    
    try:
        image = Image.open(BytesIO(frame.frame_file.read()))
        if image.mode in ["RGBA", "L"]:
            image = image.convert(mode="RGB")

        image.load()
    except Exception as e:
        logger.error(f"Image {frame.video.name} - {frame.frame_number} cannot be processed: {e}")
        frame.status = 'FAILED'
        frame.save()
        return
    finally:
        frame.frame_file.close()
    
    try:
        result = detector.generate_detections_one_image(
            image,
            f'frame{frame.frame_number:07d}.jpg',
            detection_threshold=settings.DETECTION_THRESHOLD
        )
    except Exception as e:
        logger.error(f"Image {frame.video.name} - {frame.frame_number} cannot be processed: {e}")
        frame.status = 'FAILED'
        frame.save()
        return
       
    for detection in result['detections']:
        
        x_coord, y_coord, width, height = detection['bbox']

        detection = Detection(frame=frame,
                              category=detection['category'],
                              confidence=detection['conf'],
                              x_coord=x_coord,
                              y_coord=y_coord,
                              box_width=width,
                              box_height=height)
        detection.save()
    
    frame.status = 'COMPLETED'
    frame.save()

    if settings.CLEANUP_AFTER_PROCESSING and not result['detections']:
        try:
            frame_path = frame.frame_file.path
            frame.frame_file.delete(save=True)
            # Clean up empty parent directory
            parent_dir = os.path.dirname(frame_path)
            if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        except Exception as e:
            logger.warning(f'Failed to delete frame file after detection: {e}')


@celery_app.task
def track_and_filter_detections(video_id: int) -> int:
    """
    Track detections across frames and filter out static objects.

    Uses IoU-based tracking to associate detections across frames.
    A track is considered a FALSE POSITIVE (static) if:
      - It doesn't move enough (displacement < threshold), AND
      - It's either long-duration OR low-confidence

    Real animals may stand still (e.g., deer munching grass), so we keep
    short-duration high-confidence tracks even if they don't move much.

    Args:
        video_id: The ID of the Video to process
    """
    from .tracking import SimpleTracker, group_detections_by_frame

    logger.info(f'Starting tracking for video {video_id}')

    video = Video.objects.get(id=video_id)
    frame_rate = video.frame_rate or 20.0

    # Get all detections ordered by frame
    detections = Detection.objects.filter(
        frame__video_chunk__video_id=video_id
    ).select_related('frame').order_by('frame__frame_number')

    if not detections.exists():
        logger.info(f'No detections found for video {video_id}, skipping tracking')
        return video_id

    # Group by frame number
    frames_dict = group_detections_by_frame(detections)

    # Run tracker
    tracker = SimpleTracker(
        iou_threshold=settings.TRACKING_IOU_THRESHOLD,
        max_age=settings.TRACKING_MAX_AGE,
        min_hits=3
    )

    for frame_num in sorted(frames_dict.keys()):
        frame_detections = frames_dict[frame_num]
        tracker.update(frame_detections, frame_num)

    # Finalize tracking
    tracker.finalize()

    # Create Track records and associate detections
    static_track_ids = []
    for tracked_obj in tracker.get_completed_tracks():
        displacement = tracked_obj.displacement()
        duration_seconds = (tracked_obj.end_frame - tracked_obj.start_frame) / frame_rate

        # Get max confidence for this track
        track_dets = Detection.objects.filter(id__in=tracked_obj.detection_ids)
        max_confidence = max(d.confidence for d in track_dets) if track_dets else 0.0

        # Determine if this is a real animal or static false positive
        # Rule 1: Moving tracks are real
        is_moving = displacement >= settings.MIN_DISPLACEMENT_THRESHOLD

        # Rule 2: Brief high-confidence tracks are likely real animals standing still
        # (e.g., deer munching grass for 10-15 seconds)
        is_brief_high_conf = (
            duration_seconds < settings.TRACKING_MAX_STATIC_DURATION and
            max_confidence >= settings.TRACKING_MIN_CONFIDENCE_OVERRIDE
        )

        # Keep if moving OR brief+high-confidence
        is_static = not (is_moving or is_brief_high_conf)

        track_record = Track.objects.create(
            video_id=video_id,
            track_id=tracked_obj.track_id,
            category=tracked_obj.category,
            start_frame=tracked_obj.start_frame,
            end_frame=tracked_obj.end_frame,
            total_displacement=displacement,
            is_static=is_static
        )

        # Associate detections with track
        Detection.objects.filter(id__in=tracked_obj.detection_ids).update(track=track_record)

        if is_static:
            static_track_ids.append(track_record.id)
            logger.info(
                f'Track {tracked_obj.track_id}: STATIC (disp={displacement:.4f}, '
                f'dur={duration_seconds:.1f}s, conf={max_confidence:.2f}), '
                f'{len(tracked_obj.detection_ids)} detections will be deleted'
            )
        else:
            reason = 'moving' if is_moving else 'high-confidence'
            logger.info(
                f'Track {tracked_obj.track_id}: KEEP ({reason}, disp={displacement:.4f}, '
                f'dur={duration_seconds:.1f}s, conf={max_confidence:.2f}), '
                f'{len(tracked_obj.detection_ids)} detections kept'
            )

    # Delete detections belonging to static tracks
    if static_track_ids:
        deleted_count, _ = Detection.objects.filter(track_id__in=static_track_ids).delete()
        logger.info(f'Deleted {deleted_count} detections from {len(static_track_ids)} static tracks')

    logger.info(f'Tracking complete for video {video_id}')

    return video_id


@celery_app.task
def find_completed_videos() -> None:
    """Check for videos that have finished processing and trigger output generation."""
    # Bulk query: get all non-completed videos with their chunk stats in one query
    videos_with_stats = Video.objects.exclude(status='COMPLETED').annotate(
        chunk_count=Count('videochunk'),
        incomplete_chunks=Count('videochunk', filter=~Q(videochunk__status='COMPLETED')),
        chunk_frame_total=Sum('videochunk__frame_count'),
    ).filter(
        chunk_count__gt=0,  # Has at least one chunk
        incomplete_chunks=0,  # All chunks are completed
    ).values_list('id', 'chunk_frame_total')

    # Build dict of video_id -> chunk_frame_total
    video_targets = {vid: total for vid, total in videos_with_stats}

    if not video_targets:
        return

    # Bulk query: count completed frames per video
    completed_counts = Frame.objects.filter(
        video_chunk__video_id__in=video_targets.keys(),
        status='COMPLETED'
    ).values('video_chunk__video_id').annotate(
        completed=Count('id')
    )

    completed_by_video = {
        row['video_chunk__video_id']: row['completed']
        for row in completed_counts
    }

    # Find videos that are ready to complete
    ready_video_ids = [
        vid for vid, target in video_targets.items()
        if completed_by_video.get(vid, 0) >= target
    ]

    if not ready_video_ids:
        return

    # Mark videos as completed and update frame_count with accurate value from chunks
    for video_id in ready_video_ids:
        Video.objects.filter(id=video_id).update(
            status='COMPLETED',
            frame_count=video_targets[video_id]  # Accurate count from chunks
        )

    # Trigger post-processing for each completed video
    for video_id in ready_video_ids:
        if settings.USE_SYNCHRONOUS_PROCESSING:
            if settings.USE_TRACKING:
                track_and_filter_detections(video_id)
            draw_detections(video_id)
        else:
            # Chain tracking and drawing tasks to run sequentially but async
            if settings.USE_TRACKING:
                chain(
                    track_and_filter_detections.s(video_id),
                    draw_detections.s()
                ).delay()
            else:
                draw_detections.delay(video_id)


@celery_app.task(bind=True, max_retries=None)
def draw_detections(self, video_id):

    def group_by_frame(detection):
        return detection.frame
    
    detections = Detection.objects.filter(frame__video_chunk__video_id=video_id).order_by('frame__frame_number')
    for frame, detections in itertools.groupby(detections, key=group_by_frame):
        detections_list = []

        for detection in detections:
            d = {
                'category': detection.category,
                'conf': detection.confidence,
                'bbox': [
                    detection.x_coord,
                    detection.y_coord,
                    detection.box_width,
                    detection.box_height
                ]
            }
            detections_list.append(d)
        
        frame_image = Image.open(BytesIO(frame.frame_file.read()))
        frame_image.load()
        frame.frame_file.close()

        render_detection_bounding_boxes(detections_list, frame_image)
        output_image = BytesIO()
        frame_image.save(output_image, format='JPEG')
        output_image.seek(0)
        frame.detections_file = File(output_image, f'detections{frame.frame_number:07d}.jpg')

        # Delete the original frame file - we no longer need it after drawing detections
        if settings.CLEANUP_AFTER_PROCESSING and frame.frame_file:
            try:
                frame_path = frame.frame_file.path
                frame.frame_file.delete(save=False)
                # Clean up empty parent directory
                parent_dir = os.path.dirname(frame_path)
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except Exception as e:
                logger.warning(f'Failed to delete frame file: {e}')

        try:
            frame.save()
        except OSError as e:
            self.retry(exc=e, countdown=60)

    if settings.USE_SYNCHRONOUS_PROCESSING:
        make_detection_video(video_id)
    else:
        make_detection_video.delay(video_id)


@celery_app.task(bind=True, max_retries=None)
def make_detection_video(self, video_id):
    video = Video.objects.get(id=video_id)

    output_detections_dir = os.path.join(settings.STORAGE_DIR, 'detections')
    video_basename = os.path.basename(video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_detections_dir, video_basename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    frames = Frame.objects.filter(video_chunk__video_id=video_id)\
                          .exclude(detections_file='')\
                          .order_by('frame_number')
    
    if not frames.exists():
        logger.info(f'No detections found for video {video_id}. Skipping output video creation.')
        return
    
    for index, frame in enumerate(frames):

        try:
            with open(f'{output_dir}/detection{index:07d}.jpg', 'wb') as f:
                f.write(frame.detections_file.read())
        except OSError as e:
            self.retry(exc=e, countdown=60)
        finally:
            frame.detections_file.close()


    if settings.USE_CPU_ONLY:
        # CPU-only FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-framerate',
            f'{video.frame_rate}',
            '-i',
            f'{output_dir}/detection%07d.jpg',
            '-c:v',
            'libx264',
            f'{output_dir}/{video_basename}-detections.mp4',
        ]
    else:
        # GPU-accelerated FFmpeg command (using nvenc for encoding)
        ffmpeg_cmd = [
            'ffmpeg',
            '-framerate',
            f'{video.frame_rate}',
            '-i',
            f'{output_dir}/detection%07d.jpg',
            '-c:v',
            'h264_nvenc',
            f'{output_dir}/{video_basename}-detections.mp4',
        ]
    
    subprocess.run(ffmpeg_cmd)

    with open(f'{output_dir}/{video_basename}-detections.mp4', 'rb') as f:
        video.detections_file = File(f, f'{video_basename}-detections.mp4')

        try:
            video.save()
        except OSError as e:
            self.retry(exc=e, countdown=60)

    shutil.rmtree(output_dir)

    # Clean up detection images from storage after video is created
    if settings.CLEANUP_AFTER_PROCESSING:
        for frame in frames:
            if frame.detections_file:
                try:
                    frame.detections_file.delete(save=True)
                except Exception as e:
                    logger.warning(f'Failed to delete detections file for frame {frame.id}: {e}')
