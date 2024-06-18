import os
import re
import subprocess
import logging
import math
import json
from io import BytesIO
from pathlib import Path

import numpy
import requests
from requests_toolbelt.multipart.decoder import MultipartDecoder
from PIL import Image
from decord import VideoReader, gpu
import cv2

from django.conf import settings
from django.core.files import File
from video_processor import celery_app

from .models import Video, Frame, Detection

logger = logging.getLogger(__name__)


@celery_app.task
def analyze(video_id):
    video = Video.objects.get(id=video_id)
    video_bytes = BytesIO(video.video_file.read())
    video_reader = VideoReader(video_bytes, ctx=gpu(0))
    
    frame_count = len(video_reader)
    worker_count = settings.WORKER_COUNT
    
    frame_list = numpy.array(list(range(1, frame_count)))
    chunks = [[a[0], a[-1]] for a in numpy.array_split(frame_list, worker_count)]

    for start_frame, end_frame in chunks:
        extract_frames.delay(video.id, int(start_frame), int(end_frame))


@celery_app.task
def extract_frames(video_id,
                   start_frame,
                   end_frame):
    
    video = Video.objects.get(id=video_id)

    video_bytes = BytesIO(video.video_file.read())
    video_reader = VideoReader(video_bytes, ctx=gpu(0))
    frame_batch = video_reader.get_batch(list(range(start_frame, end_frame + 1)))

    for frame_number, frame in enumerate(frame_batch.asnumpy()):
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image = BytesIO(encoded_frame)

        frame = Frame(video=video,
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
