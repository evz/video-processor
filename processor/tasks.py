import os
import re
import subprocess
import logging
import math
import json
from io import BytesIO
from pathlib import Path

import requests
from requests_toolbelt.multipart.decoder import MultipartDecoder
from PIL import Image

from django.conf import settings
from django.core.files import File
from video_processor import celery_app

from .models import Video, Frame, Detection

logger = logging.getLogger(__name__)


@celery_app.task
def add(x, y):
    return x + y


@celery_app.task
def analyze_video(video_path):
    video_info = subprocess.run(
        [
            'ffmpeg',
            '-i',
            video_path,
            '-map',
            '0:v:0',
            '-c',
            'copy',
            '-f',
            'null',
            '-',
        ],
        capture_output=True
    )
    
    output_str = video_info.stderr.decode('utf-8')
    matches = re.findall(r'frame=[\s+]*(\d+)', output_str)
    frame_count = int(matches[-1])
    
    video_name = os.path.basename(video_path)
    
    worker_count = settings.EXTRACT_WORKER_COUNT
    chunk_size = math.ceil(frame_count / worker_count)
    
    video_path = Path(video_path)
    
    with video_path.open(mode='rb') as f:
        video = Video(name=video_name,
                      video_file=File(f, name=video_path.name),
                      frame_count=frame_count,
                      status='ENQUEUED')
        video.save()
    
    video_name, _ = os.path.splitext(video.name)
    frame_output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 'frames', video_name)
    detections_output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 'detections', video_name)
    os.makedirs(frame_output_path, exist_ok=True)
    os.makedirs(detections_output_path, exist_ok=True)
    
    for worker in range(worker_count):
        extract_frames.delay(video.id, 
                             worker * chunk_size, 
                             chunk_size, 
                             frame_count)


@celery_app.task
def extract_frames(video_id, 
                   start_frame, 
                   number_of_frames_to_extract,
                   total_frames):
    
    video = Video.objects.get(id=video_id)
    video.status = 'PROCESSING'
    video.save()

    if start_frame + number_of_frames_to_extract > total_frames:
        number_of_frames_to_extract = total_frames - (start_frame - 1)
    
    video_name, _ = os.path.splitext(video.name)
    output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 'frames', video_name)
    video_output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 'frames', video.name)
    
    os.makedirs(output_path, exist_ok=True)

    with open(video_output_path, 'wb') as f:
        f.write(video.video_file.read())
        video.video_file.close()

    subprocess.run(
        [
            'ffmpeg',
            '-i',
            video_output_path,
            '-start_number',
            str(start_frame),
            '-frames:v',
            str(number_of_frames_to_extract),
            '-vf',
            'scale=1920:1080',
            f'{output_path}/frame%07d.jpg',
        ]
    )
    
    frames = []
    
    os.remove(video_output_path)

    for frame in range(start_frame, (start_frame + number_of_frames_to_extract)):
        frame_path = Path(f'{output_path}/frame{frame:07d}.jpg')
        with frame_path.open(mode='rb') as f:
            frame = Frame(video=video,
                          frame_file=File(f, name=frame_path.name),
                          frame_number=frame,
                          status='ENQUEUED')
            frame.save()
        
        detect.delay(frame.id)
        
        os.remove(frame_path)

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
    
    completed_count = Frame.objects.filter(video=frame.video).filter(status__in=['COMPLETED', 'FAILED']).count()
    
    if completed_count == frame.video.frame_count:
        frame.video.status = 'COMPLETED'
        frame.video.save()
