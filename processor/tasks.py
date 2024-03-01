import os
import re
import subprocess
import logging
import math
import json

import requests
from requests_toolbelt.multipart.decoder import MultipartDecoder

from django.conf import settings
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
    
    frame_count = int(re.findall(r'frame=\s+(\d+)', video_info.stderr.decode('utf-8'))[-1])

    video_name = os.path.basename(video_path)
    
    worker_count = settings.EXTRACT_WORKER_COUNT
    chunk_size = math.ceil(frame_count / worker_count)
    
    video = Video(name=video_name,
                  path=video_path,
                  frame_count=frame_count,
                  status='ENQUEUED')
    video.save()
    
    video_name, _ = os.path.splitext(os.path.basename(video.path))
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
    
    video_name, _ = os.path.splitext(os.path.basename(video.path))
    output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 'frames', video_name)

    subprocess.run(
        [
            'ffmpeg',
            '-i',
            video.path,
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

    for frame in range(start_frame, (start_frame + number_of_frames_to_extract)):
        frame_path = f'{output_path}/frame{frame:07d}.jpg'
        with open(frame_path, 'rb') as f:
            frame = Frame(video=video,
                          frame_data=f.read(),
                          frame_number=frame,
                          status='ENQUEUED')
            frame.save()
        detect.delay(frame.id)


@celery_app.task
def detect(frame_id):
    
    frame = Frame.objects.get(id=frame_id)
    frame.status = 'PROCESSING'
    frame.save()
    
    frame_name = f'frame{frame.frame_number:07d}.jpg'

    image = [
        ('images', (frame_name, frame.frame_data, 'image/jpeg'))
    ]
    
    detector_url = f'http://nginx:4000/v1/camera-trap/sync/detect'
    
    params = {
        'min_confidence': '0.65',
        'render': 'true',
    }

    response = requests.post(detector_url, files=image, params=params)
    
    content, image = MultipartDecoder.from_response(response).parts
    response_content = json.loads(content.content)
    
    if response_content[frame_name]:
        for detection in response_content[frame_name]:
            
            x_coord, y_coord, width, height, confidence, category = detection

            detection = Detection(frame=frame,
                                  category=category,
                                  confidence=confidence,
                                  x_coord=x_coord,
                                  y_coord=y_coord,
                                  box_width=width,
                                  box_height=height)
            detection.save()
        
        video_name, _ = os.path.splitext(os.path.basename(frame.video.path))
        
        detections_output_path = os.path.join(settings.FRAMES_OUTPUT_PATH, 
                                              'detections', 
                                              video_name)

        with open(os.path.join(detections_output_path, frame_name), 'wb') as f:
            f.write(image.content)
    
    frame.status = 'COMPLETED'
    frame.save()
    
    completed_count = Frame.objects.filter(video=frame.video).filter(status__in=['COMPLETED', 'FAILED']).count()
    
    if completed_count == frame.video.frame_count:
        frame.video.status = 'COMPLETED'
        frame.video.save()
