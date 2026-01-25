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
from celery.signals import worker_process_init, task_failure
from video_processor import celery_app

from .models import Video, VideoChunk, Frame, Detection

logger = logging.getLogger(__name__)


class DetectionResult(TypedDict):
    """Type definition for MegaDetector detection results."""
    category: str
    conf: float
    bbox: List[float]


class VideoChunkCompleted(FileSystemEventHandler):
    @staticmethod
    def on_closed(event):
        from decord import VideoReader, cpu, gpu

        filename = os.path.basename(event.src_path)
        sequence_number = int(filename.split('.')[0][-3:])
        video_name = filename.split(".")[0][:-3]

        video = Video.objects.get(name=video_name)

        # Probe the chunk file to get its frame count
        ctx = gpu(0) if not settings.USE_CPU_ONLY else cpu(0)
        vr = VideoReader(event.src_path, ctx=ctx)
        frame_count = len(vr)

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

    # Need to get the frame rate and frame count for the incoming video
    # Use decord for accurate video metadata
    from decord import VideoReader, cpu, gpu
    ctx = gpu(0) if not settings.USE_CPU_ONLY else cpu(0)
    vr = VideoReader(input_filename, ctx=ctx)
    frame_rate = vr.get_avg_fps()
    frame_count = len(vr)
    
    video.status = 'PROCESSING'
    video.name = video_basename
    video.frame_count = frame_count
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
        # Use software decoding but GPU encoding (nvenc) for reliability
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
            'h264_nvenc',
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

    if video_chunk.status != 'ENQUEUED':
        return

    video_chunk.status = 'PROCESSING'
    video_chunk.save()

    output_frames_dir = os.path.join(settings.STORAGE_DIR, 'frames')
    video_basename = os.path.basename(video_chunk.video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_frames_dir, video_basename)

    os.makedirs(output_dir, exist_ok=True)

    # Use decord to extract frames instead of ffmpeg
    # Determine device context (GPU or CPU)
    from decord import VideoReader, cpu, gpu
    ctx = gpu(0) if not settings.USE_CPU_ONLY else cpu(0)

    # Open video with decord
    vr = VideoReader(video_chunk.video_file.path, ctx=ctx)

    # Extract and save each frame
    frame_number = int(start_frame_number)
    for frame_idx in range(len(vr)):
        # Get frame as numpy array (RGB format)
        frame_array = vr[frame_idx].asnumpy()

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


@celery_app.task
def detect(frame_id: int) -> None:
    """
    Run AI detection on a single frame to find animals, people, or vehicles.
    
    Uses MegaDetector v5a model to analyze the frame and store detection results.
    
    Args:
        frame_id: The ID of the Frame model instance to analyze
    """
    
    frame = Frame.objects.get(id=frame_id)
    
    if frame.status != 'ENQUEUED':
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
            detection_threshold=0.65
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
        os.remove(frame.frame_file.path)


@celery_app.task
def find_completed_videos() -> None:
    """Check for videos that have finished processing and trigger output generation."""
    for video in Video.objects.exclude(status='COMPLETED'):
        completed_frames = Frame.objects.filter(video_chunk__video_id=video.id).filter(status='COMPLETED').count()
        
        if video.frame_count:

            # Sometimes we can end up with more frames extracted from the video
            # than what decord thought there were to begin with. I really don't
            # understand why but, that's why we have to compare things this way ...
            if completed_frames >= video.frame_count:
                video.status = 'COMPLETED'
                video.save()
                
                if settings.USE_SYNCHRONOUS_PROCESSING:
                    draw_detections(video.id)
                else:
                    draw_detections.delay(video.id)


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
