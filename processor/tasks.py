import os
import subprocess
import logging
import math
import json
import time
import itertools
import shutil

from io import BytesIO
from pathlib import Path
from threading import Thread, Event

import numpy
from PIL import Image
from decord import VideoReader, gpu
import cv2
from megadetector.visualization.visualization_utils import render_detection_bounding_boxes

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
        task = self._args[2]

        completed = []

        while True:
            time.sleep(2.5)
            processing_files = list(os.listdir(output_path))

            for filename in processing_files:
                modified_time = os.path.getmtime(f'{output_path}/{filename}')
                
                if (time.time() - modified_time) > 1 and filename not in completed:
                    
                    sequence_number = int(filename.split('.')[0][-3:])

                    with open(f'{output_path}/{filename}', 'rb') as f:
                        video_chunk = VideoChunk(name=filename,
                                                 video=video,
                                                 video_file=File(f, filename),
                                                 sequence_number=sequence_number)

                        try:
                            video_chunk.save()
                        except OSError as e:
                            task.retry(exc=e, countdown=60)

                    completed.append(filename)
                    
                    extract_frames.delay(video_chunk.id)
                    
                    os.remove(f'{output_path}/{filename}')

            if self.stopped():
                break


@celery_app.task(bind=True)
def chunk_video(self, video_id):
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
    
    video_bytes = BytesIO(video.video_file.read())
   
    try:
        with open(input_filename, 'wb') as f:
            video_reader = VideoReader(video_bytes, ctx=gpu(0))
            video_bytes.seek(0)
            f.write(video_bytes.read())
    except OSError as e:
        self.retry(exc=e, countdown=60)
    
    # Need to get the frame rate for the incoming video
    video_cap = cv2.VideoCapture(input_filename)
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()

    video.status = 'PROCESSING'
    video.name = video_basename
    video.frame_count = len(video_reader)
    video.frame_rate = frame_rate
    video.save()

    monitor_thread = MonitorThread(args=[output_dir, video_id, self])
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
        '00:05:00', # This is somewhat arbitrary but it works
        '-f',
        'segment',
        '-reset_timestamps',
        '1', 
        f'{output_dir}/{video_basename}%03d.mp4'
    ])

    monitor_thread.stop()
    monitor_thread.join()
    
    # Sometimes there's a chunk of the video left that didn't get enqueued.
    # Make sure to enqueue everything.
    processing_files = list(os.listdir(output_dir))

    for filename in processing_files:
        
        with open(f'{output_dir}/{filename}', 'rb') as f:
            video_chunk = VideoChunk(name=filename,
                                     video=video,
                                     video_file=File(f, filename))
            try:
                video_chunk.save()
            except OSError as e:
                self.retry(exc=e, countdown=10)

        extract_frames.delay(video_chunk.id)
        os.remove(f'{output_dir}/{filename}')

    os.remove(input_filename)


@celery_app.task(bind=True)
def extract_frames(self, video_chunk_id):
    
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
    
    video_bytes = BytesIO(video_chunk.video_file.read())
    video_reader = VideoReader(video_bytes, ctx=gpu(0))

    # Each chunk is 5 minutes. Multiply that by the frame rate of the video and
    # the sequence number of the video chunk to get the starting frame number
    # for the chunk. Add one because we don't want to have the frame numbers be
    # zero indexed.
    start_frame_number = (video_chunk.sequence_number * video_chunk.video.frame_rate * 5 * 60) + 1

    for frame_number, frame in enumerate(video_reader, start=start_frame_number):
        scaled = cv2.resize(frame.asnumpy(), (1920, 1080))
        _, encoded_frame = cv2.imencode('.jpg', scaled, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image = BytesIO(encoded_frame)

        frame = Frame(video_chunk=video_chunk,
                      frame_file=File(image, f'frame{frame_number:07d}.jpg'),
                      detections_file=None,
                      frame_number=frame_number,
                      status='ENQUEUED')
        try:
            frame.save()
        except OSError as e:
            self.retry(exc=e, countdown=60)
        
        detect.delay(frame.id)
    
    video_chunk.status = 'COMPLETED'
    video_chunk.save()
    
    if settings.CLEANUP_AFTER_PROCESSING:
        os.remove(video_chunk.video_file.path)


detector = None

@celery_app.task
def detect(frame_id):
    
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
def find_completed_videos():
    for video in Video.objects.exclude(status='COMPLETED'):
        completed_frames = Frame.objects.filter(video_chunk__video_id=video.id).filter(status='COMPLETED').count()
        
        if video.frame_count:

            # Sometimes we can end up with more frames extracted from the video
            # than what decord thought there were to begin with. I really don't
            # understand why but, that's why we have to compare things this way ...
            if completed_frames >= video.frame_count:
                video.status = 'COMPLETED'
                video.save()
                
                draw_detections.delay(video.id)


@celery_app.task(bind=True)
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

        render_detection_bounding_boxes(detections_list, frame_image)
        output_image = BytesIO()
        frame_image.save(output_image, format='JPEG')
        output_image.seek(0)
        frame.detections_file = File(output_image, f'detections{frame.frame_number:07d}.jpg')
        
        try:
            frame.save()
        except OSError as e:
            self.retry(exc=e, countdown=60)

    make_detection_video.delay(video_id)


@celery_app.task(bind=True)
def make_detection_video(self, video_id):
    video = Video.objects.get(id=video_id)

    output_detections_dir = os.path.join(settings.STORAGE_DIR, 'detections')
    video_basename = os.path.basename(video.video_file.name).split('.')[0]
    output_dir = os.path.join(output_detections_dir, video_basename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    frames = Frame.objects.filter(video_chunk__video_id=video_id)\
                          .exclude(detections_file='')\
                          .order_by('frame_number')
    
    for index, frame in enumerate(frames):

        try:
            with open(f'{output_dir}/detection{index:07d}.jpg', 'wb') as f:
                f.write(frame.detections_file.read())
        except OSError as e:
            self.retry(exc=e, countdown=60)


    subprocess.run([
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
    ])

    with open(f'{output_dir}/{video_basename}-detections.mp4', 'rb') as f:
        video.detections_file = File(f, f'{video_basename}-detections.mp4')

        try:
            video.save()
        except OSError as e:
            self.retry(exc=e, countdown=60)

    shutil.rmtree(output_dir)
