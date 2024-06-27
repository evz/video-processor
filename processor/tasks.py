import os
import re
import subprocess
import logging
import math
import json
from io import BytesIO
from pathlib import Path
import time
from threading import Thread, Event

import numpy
import requests
from requests_toolbelt.multipart.decoder import MultipartDecoder
from PIL import Image
from decord import VideoReader, gpu
import cv2

from django.conf import settings
from django.core.files import File
from video_processor import celery_app

from .models import Video, VideoChunk, Frame, Detection

logger = logging.getLogger(__name__)


class MonitorThread(Thread):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_event = Event()

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.is_set()

    def run(self):
        
        logger.info(self._args)

        output_path = self._args[0]
        video = Video.objects.get(id=self._args[1])
        
        completed = []

        while True:
            time.sleep(2.5)
            processing_files = list(os.listdir(output_path))

            for filename in processing_files:
                modified_time = os.path.getmtime(f'{output_path}/{filename}')
                
                if (time.time() - modified_time) > 1 and filename not in completed:

                    with open(f'{output_path}/{filename}', 'rb') as f:
                        video_chunk = VideoChunk(name=filename,
                                                 video=video,
                                                 video_file=File(f, filename))
                        video_chunk.save()

                    completed.append(filename)
                    
                    extract_frames.delay(video_chunk.id)
                    
                    os.remove(f'{output_path}/{filename}')

            if self.stopped():
                break


@celery_app.task
def chunk_video(video_id):
    # It looks like there's a chance that the message gets picked up by a
    # worker before the row gets saved to the DB, I guess. This is somewhat
    # maddening since this gets triggered by the `post_save` hook but,
    # whatever. Now you know why this retry crap is here.
    retries = 0
    logger.info(f'received {video_id}')
    
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
    
    input_video_dir = os.path.join(settings.STORAGE_DIR, 'input_videos')
    output_video_dir = os.path.join(settings.STORAGE_DIR, 'output_videos')
    video_basename = os.path.basename(video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_video_dir, video_basename)
    
    os.makedirs(input_video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_filename = os.path.join(input_video_dir, video.video_file.name)
    
    with open(input_filename, 'wb') as f:
        f.write(video.video_file.read())
    
    monitor_thread = MonitorThread(args=[output_dir, video_id])
    monitor_thread.start()

    subprocess.run([
        'ffmpeg',
        '-hwaccel',
        'cuda',
        '-i', 
        f'{input_filename}',
        '-c:v',
        'h264_nvenc',
        '-map',
        '0',
        '-segment_time',
        '00:05:00',
        '-f',
        'segment',
        '-reset_timestamps',
        '1', 
        f'{output_dir}/{video_basename}%03d.mp4'
    ])

    monitor_thread.stop()
    monitor_thread.join()
    
    os.remove(input_filename)


@celery_app.task
def extract_frames(video_chunk_id):
    
    # It looks like there's a chance that the message gets picked up by a
    # worker before the row gets saved to the DB, I guess. This is somewhat
    # maddening since this gets triggered by the `post_save` hook but,
    # whatever. Now you know why this retry crap is here.
    retries = 0
    
    while retries <= 5:
        try:
            video = VideoChunk.objects.get(id=video_chunk_id)
            break
        except VideoChunk.DoesNotExist as e:
            logger.info(f'video {video_chunk_id} is not here I guess')
            if retries == 5:
                raise e
            retries += 1
            time.sleep(0.1)

    logger.info(f'found video {video_chunk_id}')
    video_bytes = BytesIO(video.video_file.read())
    video_reader = VideoReader(video_bytes, ctx=gpu(0))
    
    for frame_number, frame in enumerate(video_reader):
        scaled = cv2.resize(frame.asnumpy(), (1920, 1080))
        _, encoded_frame = cv2.imencode('.jpg', scaled, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image = BytesIO(encoded_frame)

        frame = Frame(video_chunk=video,
                      frame_file=File(image, f'frame{frame_number:07d}.jpg'),
                      frame_number=frame_number,
                      status='ENQUEUED')
        frame.save()
        
        detect.delay(frame.id)


detector = None

@celery_app.task
def detect(frame_id):

    global detector 
    
    if not detector:
        from detection.run_detector import load_detector
        detector = load_detector('MDV5A')

    frame = Frame.objects.get(id=frame_id)
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
