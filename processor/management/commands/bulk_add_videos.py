"""
Management command to bulk add videos from a directory for processing.

Usage:
    python manage.py bulk_add_videos /path/to/videos
    python manage.py bulk_add_videos /path/to/videos --pattern "*.mp4"
    python manage.py bulk_add_videos /path/to/videos --recursive
"""

import os
import glob
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from processor.models import Video


class Command(BaseCommand):
    help = 'Bulk add videos from a directory for processing'

    def add_arguments(self, parser):
        parser.add_argument(
            'directory',
            type=str,
            help='Directory containing videos to process'
        )
        parser.add_argument(
            '--pattern',
            type=str,
            default='*.{mp4,avi,mov,mkv,MP4,AVI,MOV,MKV}',
            help='Glob pattern for video files (default: *.{mp4,avi,mov,mkv,MP4,AVI,MOV,MKV})'
        )
        parser.add_argument(
            '--recursive',
            action='store_true',
            help='Recursively search subdirectories'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be processed without actually adding videos'
        )

    def handle(self, *args, **options):
        directory = options['directory']
        pattern = options['pattern']
        recursive = options['recursive']
        dry_run = options['dry_run']

        # Validate directory
        if not os.path.isdir(directory):
            raise CommandError(f'Directory not found: {directory}')

        # Find video files
        video_files = self.find_videos(directory, pattern, recursive)

        if not video_files:
            self.stdout.write(self.style.WARNING(
                f'No video files found in {directory} matching pattern {pattern}'
            ))
            return

        self.stdout.write(self.style.SUCCESS(
            f'Found {len(video_files)} video file(s)'
        ))

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No videos will be added\n'))
            for video_path in video_files:
                self.stdout.write(f'  Would process: {video_path}')
            return

        # Process each video
        added_count = 0
        skipped_count = 0
        error_count = 0

        for video_path in video_files:
            try:
                result = self.add_video(video_path)
                if result == 'added':
                    added_count += 1
                    self.stdout.write(self.style.SUCCESS(f'✓ Added: {video_path}'))
                elif result == 'skipped':
                    skipped_count += 1
                    self.stdout.write(self.style.WARNING(f'⊘ Skipped (already exists): {video_path}'))
            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'✗ Error adding {video_path}: {e}'))

        # Summary
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS(f'Added:   {added_count}'))
        if skipped_count > 0:
            self.stdout.write(self.style.WARNING(f'Skipped: {skipped_count}'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'Errors:  {error_count}'))
        self.stdout.write('=' * 60 + '\n')

        if added_count > 0:
            self.stdout.write(self.style.SUCCESS(
                f'\n{added_count} video(s) queued for processing!'
            ))

    def find_videos(self, directory, pattern, recursive):
        """Find all video files matching the pattern."""
        video_files = []

        # Handle brace expansion patterns like *.{mp4,avi}
        if '{' in pattern and '}' in pattern:
            # Extract extensions from pattern like "*.{mp4,avi,mov}"
            import re
            match = re.search(r'\{([^}]+)\}', pattern)
            if match:
                extensions = match.group(1).split(',')
                prefix = pattern[:pattern.index('{')]
                patterns = [f"{prefix}{ext}" for ext in extensions]
        else:
            patterns = [pattern]

        for pat in patterns:
            if recursive:
                search_pattern = os.path.join(directory, '**', pat)
                files = glob.glob(search_pattern, recursive=True)
            else:
                search_pattern = os.path.join(directory, pat)
                files = glob.glob(search_pattern)

            video_files.extend(files)

        # Sort for consistent ordering
        return sorted(set(video_files))

    def add_video(self, video_path):
        """Add a single video to the database."""
        filename = os.path.basename(video_path)

        # Check if video already exists
        if Video.objects.filter(video_file__endswith=filename).exists():
            return 'skipped'

        # Create Video record
        with open(video_path, 'rb') as f:
            video = Video(
                status='ENQUEUED'
            )
            video.video_file.save(filename, File(f), save=True)

        return 'added'
