import os
import subprocess
import logging
import math
import json
import time
import itertools
import shutil
import glob

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
        filename = os.path.basename(event.src_path)
        sequence_number = int(filename.split('.')[0][-3:])
        video_name = filename.split(".")[0][:-3]

        video = Video.objects.get(name=video_name)

        with open(event.src_path, 'rb') as f:
            video_chunk = VideoChunk(name=filename,
                                     video=video,
                                     video_file=File(f, filename),
                                     sequence_number=sequence_number)

            video_chunk.save()

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
    
    input_video_dir = os.path.join(settings.STORAGE_DIR, 'videos')
    output_video_dir = os.path.join(settings.STORAGE_DIR, 'output_videos')
    video_basename = os.path.basename(video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_video_dir, video_basename)
    
    os.makedirs(input_video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.join(input_video_dir, video.video_file.name)
    
    # Need to get the frame rate for the incoming video
    video_cap = cv2.VideoCapture(input_filename)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_cap.release()
    
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
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', 
            f'{input_filename}',
            '-vf',
            'scale=w=1920:h=1080',
            '-c:v',
            'libx264',
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
        ffmpeg_cmd = [
            'ffmpeg',
            '-hwaccel',
            'cuda',
            '-hwaccel_output_format',
            'cuda',
            '-i', 
            f'{input_filename}',
            '-init_hw_device',
            'cuda',
            '-vf',
            'scale_cuda=w=1920:h=1080',
            '-c:v',
            'h264_nvenc',
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
            
            with open(chunk_file, 'rb') as f:
                video_chunk = VideoChunk(name=filename,
                                       video=video,
                                       video_file=File(f, filename),
                                       sequence_number=sequence_number)
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
    
    if video_chunk.status != 'ENQUEUED':
        return

    video_chunk.status = 'PROCESSING'
    video_chunk.save()
    
    # Each chunk is 30 seconds. Multiply that by the frame rate of the video and
    # the sequence number of the video chunk to get the starting frame number
    # for the chunk. Add one because we don't want to have the frame numbers be
    # zero indexed.
    frame_count = video_chunk.video.frame_rate * 30
    start_frame_number = (video_chunk.sequence_number * frame_count) + 1
    output_frames_dir = os.path.join(settings.STORAGE_DIR, 'frames')
    video_basename = os.path.basename(video_chunk.video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_frames_dir, video_basename)

    os.makedirs(output_dir, exist_ok=True)

    subprocess.run([
        'ffmpeg',
        '-i', 
        video_chunk.video_file.path,
        '-start_number',
        str(start_frame_number),
        f'{output_dir}/frame-%07d.jpg'
    ], check=True)

    # Count actual frames extracted for this chunk instead of assuming frame_count
    frame_number = start_frame_number
    while True:
        frame_path = os.path.join(output_dir, f'frame-{frame_number:07d}.jpg')
        if not os.path.exists(frame_path):
            break
            
        with open(frame_path, 'rb') as image:
            frame = Frame(video_chunk=video_chunk,
                          frame_file=File(image, f'frame{frame_number:07d}.jpg'),
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
        # GPU-accelerated FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-hwaccel',
            'cuda',
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
