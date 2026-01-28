"""
Management command to clean up orphaned frame files.

Usage:
    python manage.py cleanup_frames --dry-run    # Show what would be deleted
    python manage.py cleanup_frames              # Delete orphaned files
"""

import os
import glob
from django.conf import settings
from django.core.management.base import BaseCommand
from processor.models import Video, Frame


class Command(BaseCommand):
    help = 'Clean up orphaned frame files from completed videos'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        frames_dir = os.path.join(settings.STORAGE_DIR, 'frames')

        if not os.path.exists(frames_dir):
            self.stdout.write(self.style.WARNING(f'Frames directory not found: {frames_dir}'))
            return

        # Get all completed videos
        completed_videos = Video.objects.filter(status='COMPLETED').values_list('name', flat=True)
        completed_video_names = set(completed_videos)

        self.stdout.write(f'Found {len(completed_video_names)} completed videos')

        # Find all frame directories
        deleted_files = 0
        deleted_dirs = 0
        total_bytes = 0

        for video_dir in glob.glob(os.path.join(frames_dir, '*')):
            if not os.path.isdir(video_dir):
                continue

            dir_name = os.path.basename(video_dir)

            # Check if this directory belongs to a completed video
            # Directory names may have chunk suffixes (e.g., ch02_20230711050652000)
            # while video names don't (e.g., ch02_20230711050652)
            is_completed = False
            for video_name in completed_video_names:
                if dir_name == video_name or dir_name.startswith(video_name):
                    is_completed = True
                    break

            if is_completed:
                # Delete jpg files in the directory
                frame_files = glob.glob(os.path.join(video_dir, '*.jpg'))
                for frame_file in frame_files:
                    file_size = os.path.getsize(frame_file)
                    total_bytes += file_size
                    deleted_files += 1
                    if not dry_run:
                        os.remove(frame_file)

                # Also delete jpg files in detections subdirectory
                detections_dir = os.path.join(video_dir, 'detections')
                if os.path.isdir(detections_dir):
                    detection_files = glob.glob(os.path.join(detections_dir, '*.jpg'))
                    for detection_file in detection_files:
                        file_size = os.path.getsize(detection_file)
                        total_bytes += file_size
                        deleted_files += 1
                        if not dry_run:
                            os.remove(detection_file)

                    # Remove detections subdirectory if empty
                    if not dry_run:
                        try:
                            if not os.listdir(detections_dir):
                                os.rmdir(detections_dir)
                        except OSError:
                            pass

                # Remove video directory if empty
                if not dry_run:
                    try:
                        if not os.listdir(video_dir):
                            os.rmdir(video_dir)
                            deleted_dirs += 1
                    except OSError:
                        pass
                else:
                    deleted_dirs += 1

        # Also clean up frame_file references in database for completed videos
        if not dry_run:
            # Clear frame_file field for frames that have been processed
            frames_updated = Frame.objects.filter(
                video_chunk__video__status='COMPLETED',
            ).exclude(frame_file='').update(frame_file='')
            self.stdout.write(f'Cleared {frames_updated} frame_file references in database')

            # Clear detections_file field for frames where video is completed
            detections_updated = Frame.objects.filter(
                video_chunk__video__status='COMPLETED'
            ).exclude(detections_file='').update(detections_file='')
            self.stdout.write(f'Cleared {detections_updated} detections_file references in database')

        # Summary
        total_mb = total_bytes / (1024 * 1024)
        total_gb = total_bytes / (1024 * 1024 * 1024)

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No files deleted\n'))

        self.stdout.write('=' * 60)
        self.stdout.write(f'Files:       {deleted_files:,}')
        self.stdout.write(f'Directories: {deleted_dirs:,}')
        self.stdout.write(f'Space:       {total_gb:.2f} GB ({total_mb:,.0f} MB)')
        self.stdout.write('=' * 60)

        if deleted_files > 0 and not dry_run:
            self.stdout.write(self.style.SUCCESS(f'\nCleaned up {deleted_files:,} orphaned frame files!'))
