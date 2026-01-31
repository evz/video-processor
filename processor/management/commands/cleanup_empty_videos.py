"""
Management command to clean up completed videos that have no detections.

Usage:
    python manage.py cleanup_empty_videos --dry-run
    python manage.py cleanup_empty_videos
"""

import os

from django.core.management.base import BaseCommand
from django.db.models import Count
from processor.models import Video, VideoChunk, Frame, Detection, Track


def format_bytes(num_bytes):
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f'{num_bytes:.1f} {unit}'
        num_bytes /= 1024.0
    return f'{num_bytes:.1f} PB'


def get_file_size(file_field):
    """Get file size in bytes, or 0 if file doesn't exist."""
    if not file_field:
        return 0
    try:
        return file_field.size
    except (FileNotFoundError, OSError):
        return 0


def file_exists(file_field):
    """Check if the file actually exists on disk."""
    if not file_field:
        return False
    try:
        return os.path.exists(file_field.path)
    except (ValueError, AttributeError):
        return False


class Command(BaseCommand):
    help = 'Delete completed videos that have no detections'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be deleted without actually deleting'
        )
        parser.add_argument(
            '--min-detections',
            type=int,
            default=0,
            help='Delete videos with fewer than this many detections (default: 0, meaning only videos with zero detections)'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        min_detections = options['min_detections']

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No changes will be made\n'))

        # Find completed videos with detection count below threshold
        videos_with_counts = Video.objects.filter(
            status='COMPLETED'
        ).annotate(
            detection_count=Count('videochunk__frame__detection')
        ).filter(
            detection_count__lte=min_detections
        )

        total_videos = videos_with_counts.count()

        if total_videos == 0:
            if min_detections == 0:
                self.stdout.write(self.style.SUCCESS('No completed videos found with zero detections.'))
            else:
                self.stdout.write(self.style.SUCCESS(
                    f'No completed videos found with {min_detections} or fewer detections.'
                ))
            return

        if min_detections == 0:
            self.stdout.write(f'Found {total_videos} video(s) with no detections\n')
        else:
            self.stdout.write(f'Found {total_videos} video(s) with {min_detections} or fewer detections\n')

        # Totals
        deleted_videos = 0
        total_files_deleted = 0
        total_bytes_freed = 0
        total_db_records = 0  # chunks + frames + detections

        for video in videos_with_counts:
            self.stdout.write(f'Video {video.id} ({video.name}): {video.detection_count} detections')

            # Count related objects
            chunks = VideoChunk.objects.filter(video=video)
            frames = Frame.objects.filter(video_chunk__video=video)
            detections = Detection.objects.filter(frame__video_chunk__video=video)
            chunk_count = chunks.count()
            frame_count = frames.count()
            detection_count = detections.count()

            # Calculate actual files and bytes that exist on disk
            files_on_disk = 0
            bytes_on_disk = 0

            # Check video file
            if file_exists(video.video_file):
                files_on_disk += 1
                bytes_on_disk += get_file_size(video.video_file)

            # Check detections video file
            if file_exists(video.detections_file):
                files_on_disk += 1
                bytes_on_disk += get_file_size(video.detections_file)

            # Check chunk files
            for chunk in chunks:
                if file_exists(chunk.video_file):
                    files_on_disk += 1
                    bytes_on_disk += get_file_size(chunk.video_file)

            # Check frame files
            for frame in frames:
                if file_exists(frame.frame_file):
                    files_on_disk += 1
                    bytes_on_disk += get_file_size(frame.frame_file)
                if file_exists(frame.detections_file):
                    files_on_disk += 1
                    bytes_on_disk += get_file_size(frame.detections_file)

            db_records = chunk_count + frame_count + detection_count

            self.stdout.write(f'  - {chunk_count} chunks, {frame_count} frames, {detection_count} detections (DB records)')
            self.stdout.write(f'  - {files_on_disk} files on disk ({format_bytes(bytes_on_disk)})')

            if not dry_run:
                # Delete files that exist
                files_deleted = 0

                # Delete video file
                if file_exists(video.video_file):
                    try:
                        video.video_file.delete(save=False)
                        files_deleted += 1
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f'  - Could not delete video file: {e}'))

                # Delete detections file
                if file_exists(video.detections_file):
                    try:
                        video.detections_file.delete(save=False)
                        files_deleted += 1
                    except Exception as e:
                        self.stdout.write(self.style.WARNING(f'  - Could not delete detections file: {e}'))

                # Delete chunk video files
                for chunk in chunks:
                    if file_exists(chunk.video_file):
                        try:
                            chunk.video_file.delete(save=False)
                            files_deleted += 1
                        except Exception as e:
                            pass

                # Delete frame files
                for frame in frames:
                    if file_exists(frame.frame_file):
                        try:
                            frame.frame_file.delete(save=False)
                            files_deleted += 1
                        except Exception as e:
                            pass
                    if file_exists(frame.detections_file):
                        try:
                            frame.detections_file.delete(save=False)
                            files_deleted += 1
                        except Exception as e:
                            pass

                # Delete related database records (tracks, chunks cascade to frames/detections)
                Track.objects.filter(video=video).delete()
                chunks.delete()  # Cascades to frames and detections

                # Clear file fields on video and mark as deleted (soft delete)
                video.video_file = ''
                video.detections_file = None
                video.status = 'DELETED'
                video.save()

                total_files_deleted += files_deleted
                total_bytes_freed += bytes_on_disk
                total_db_records += db_records
                deleted_videos += 1

                self.stdout.write(self.style.SUCCESS(f'  - Marked as deleted'))
            else:
                total_files_deleted += files_on_disk
                total_bytes_freed += bytes_on_disk
                total_db_records += db_records

        # Summary
        self.stdout.write('\n' + '=' * 60)
        if dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN - Would clean up:'))
            self.stdout.write(f'  - {total_videos} videos marked as DELETED')
            self.stdout.write(f'  - {total_db_records} related DB records (chunks, frames, detections)')
            self.stdout.write(f'  - {total_files_deleted} files on disk')
            self.stdout.write(f'  - {format_bytes(total_bytes_freed)} disk space')
        else:
            self.stdout.write(self.style.SUCCESS(f'Cleaned up {deleted_videos} videos'))
            self.stdout.write(f'  - {deleted_videos} videos marked as DELETED (records kept for deduplication)')
            self.stdout.write(f'  - {total_db_records} related DB records removed')
            self.stdout.write(f'  - {total_files_deleted} files removed from disk')
            self.stdout.write(f'  - {format_bytes(total_bytes_freed)} disk space freed')
        self.stdout.write('=' * 60 + '\n')
