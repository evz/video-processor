"""
Management command to bulk add videos from a directory for processing.

Usage:
    python manage.py bulk_add_videos /path/to/videos
    python manage.py bulk_add_videos /path/to/videos --pattern "*.mp4"
    python manage.py bulk_add_videos /path/to/videos --recursive

Time-based prioritization (requires filenames like ch<NN>_<YYYYMMDD><HHMMSS>.mp4):
    # Night-first (astronomical - requires location)
    python manage.py bulk_add_videos /path/to/videos --priority night --lat 43.23 --lon -87.95

    # Day-first (astronomical - requires location)
    python manage.py bulk_add_videos /path/to/videos --priority day --lat 43.23 --lon -87.95

    # Specific hour range (e.g., 6 PM to 6 AM)
    python manage.py bulk_add_videos /path/to/videos --priority hours --hours 18-6

    # Only include videos in time range (skip others entirely)
    python manage.py bulk_add_videos /path/to/videos --priority night --lat 43.23 --lon -87.95 --only
"""

import os
import re
import glob
from datetime import datetime
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from processor.models import Video

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

try:
    from astral import LocationInfo
    from astral.sun import sun
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False


# Regex pattern for ch<channel>_<YYYYMMDD><HHMMSS>.mp4
FILENAME_PATTERN = re.compile(r'^ch(\d+)_(\d{8})(\d{6})\.mp4$')


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
        # Time-based prioritization options
        parser.add_argument(
            '--priority',
            type=str,
            choices=['night', 'day', 'hours', 'none'],
            default='none',
            help='Prioritize videos by time: night (after dark first), day (daylight first), hours (specific range first), none (no priority)'
        )
        parser.add_argument(
            '--lat',
            type=float,
            help='Latitude for astronomical calculations (required for night/day priority)'
        )
        parser.add_argument(
            '--lon',
            type=float,
            help='Longitude for astronomical calculations (required for night/day priority)'
        )
        parser.add_argument(
            '--timezone',
            type=str,
            default='America/Chicago',
            help='Timezone for time calculations (default: America/Chicago)'
        )
        parser.add_argument(
            '--hours',
            type=str,
            help='Hour range for hours priority, e.g., "18-6" for 6PM-6AM, "9-17" for 9AM-5PM'
        )
        parser.add_argument(
            '--only',
            action='store_true',
            help='Only include priority videos, skip non-matching videos entirely'
        )
        parser.add_argument(
            '--limit',
            type=int,
            help='Maximum number of videos to add'
        )

    def handle(self, *args, **options):
        directory = options['directory']
        pattern = options['pattern']
        recursive = options['recursive']
        dry_run = options['dry_run']
        priority = options['priority']
        limit = options['limit']

        # Validate directory
        if not os.path.isdir(directory):
            raise CommandError(f'Directory not found: {directory}')

        # Validate priority options
        if priority in ['night', 'day']:
            if not ASTRAL_AVAILABLE:
                raise CommandError(
                    f'The astral library is required for {priority} priority. '
                    'Install it with: pip install astral'
                )
            if options['lat'] is None or options['lon'] is None:
                raise CommandError(f'--lat and --lon are required for {priority} priority')

        if priority == 'hours':
            if not options['hours']:
                raise CommandError('--hours is required for hours priority (e.g., --hours 18-6)')
            start_hour, end_hour = self.parse_hours(options['hours'])
        else:
            start_hour, end_hour = None, None

        # Setup timezone and location
        try:
            tz = ZoneInfo(options['timezone'])
        except Exception as e:
            raise CommandError(f'Invalid timezone "{options["timezone"]}": {e}')

        location = None
        if priority in ['night', 'day']:
            location = LocationInfo(
                name="Camera Location",
                region="",
                timezone=options['timezone'],
                latitude=options['lat'],
                longitude=options['lon']
            )
            self.stdout.write(f'Location: {options["lat"]}, {options["lon"]} ({options["timezone"]})')

        # Find video files
        video_files = self.find_videos(directory, pattern, recursive)

        if not video_files:
            self.stdout.write(self.style.WARNING(
                f'No video files found in {directory} matching pattern {pattern}'
            ))
            return

        self.stdout.write(f'Found {len(video_files)} video file(s)')

        # Apply time-based sorting/filtering if requested
        if priority != 'none':
            video_files = self.apply_priority(
                video_files, priority, location, tz,
                start_hour, end_hour, options['only']
            )

            if not video_files:
                self.stdout.write(self.style.WARNING('No videos match the time criteria'))
                return

        # Apply limit
        if limit and len(video_files) > limit:
            video_files = video_files[:limit]
            self.stdout.write(f'Limited to {limit} videos')

        self.stdout.write(self.style.SUCCESS(f'Will process {len(video_files)} video(s)'))

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No videos will be added\n'))
            for i, video_path in enumerate(video_files[:30]):
                self.stdout.write(f'  {i+1}. {os.path.basename(video_path)}')
            if len(video_files) > 30:
                self.stdout.write(f'  ... and {len(video_files) - 30} more')
            return

        # Process each video
        added_count = 0
        skipped_count = 0
        error_count = 0

        for i, video_path in enumerate(video_files):
            try:
                result = self.add_video(video_path)
                if result == 'added':
                    added_count += 1
                    if added_count % 100 == 0:
                        self.stdout.write(f'  Added {added_count} videos...')
                elif result == 'skipped':
                    skipped_count += 1
            except Exception as e:
                error_count += 1
                self.stdout.write(self.style.ERROR(f'Error adding {video_path}: {e}'))

        # Summary
        self.stdout.write('\n' + '=' * 60)
        self.stdout.write(self.style.SUCCESS(f'Added:   {added_count}'))
        if skipped_count > 0:
            self.stdout.write(self.style.WARNING(f'Skipped: {skipped_count} (already exist)'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'Errors:  {error_count}'))
        self.stdout.write('=' * 60 + '\n')

        if added_count > 0:
            self.stdout.write(self.style.SUCCESS(
                f'{added_count} video(s) queued for processing!'
            ))

    def parse_hours(self, hours_str):
        """Parse hour range like '18-6' into (start_hour, end_hour)."""
        try:
            parts = hours_str.split('-')
            if len(parts) != 2:
                raise ValueError()
            start = int(parts[0])
            end = int(parts[1])
            if not (0 <= start <= 23 and 0 <= end <= 23):
                raise ValueError()
            return start, end
        except ValueError:
            raise CommandError(
                f'Invalid hours format "{hours_str}". Use format like "18-6" (6PM-6AM) or "9-17" (9AM-5PM)'
            )

    def is_in_hour_range(self, hour, start_hour, end_hour):
        """Check if hour is within range, handling overnight ranges like 18-6."""
        if start_hour <= end_hour:
            # Normal range like 9-17
            return start_hour <= hour <= end_hour
        else:
            # Overnight range like 18-6
            return hour >= start_hour or hour <= end_hour

    def apply_priority(self, video_files, priority, location, tz, start_hour, end_hour, only_matching):
        """Sort and optionally filter videos based on time priority."""

        # Parse timestamps from filenames
        parsed_videos = []
        unparseable = []

        for filepath in video_files:
            filename = os.path.basename(filepath)
            parsed = self.parse_filename(filename)
            if parsed:
                channel, date_str, time_str = parsed
                video_dt = self.parse_datetime(date_str, time_str, tz)
                parsed_videos.append((filepath, video_dt))
            else:
                unparseable.append(filepath)

        if unparseable:
            self.stdout.write(self.style.WARNING(
                f'  {len(unparseable)} files skipped (filename not in ch<NN>_<YYYYMMDD><HHMMSS>.mp4 format)'
            ))

        if not parsed_videos:
            return []

        # Classify videos
        priority_videos = []
        other_videos = []
        sun_cache = {}

        for filepath, video_dt in parsed_videos:
            is_priority = False

            if priority == 'hours':
                is_priority = self.is_in_hour_range(video_dt.hour, start_hour, end_hour)

            elif priority in ['night', 'day']:
                date_key = video_dt.date()
                if date_key not in sun_cache:
                    try:
                        sun_times = sun(location.observer, date=date_key, tzinfo=tz)
                        sun_cache[date_key] = sun_times
                    except Exception:
                        # Default to treating as night if we can't calculate
                        sun_cache[date_key] = None

                sun_times = sun_cache[date_key]
                if sun_times:
                    sunrise = sun_times['sunrise']
                    sunset = sun_times['sunset']
                    is_night = video_dt < sunrise or video_dt > sunset
                    is_priority = is_night if priority == 'night' else not is_night
                else:
                    # Couldn't calculate sun times, treat as priority
                    is_priority = True

            if is_priority:
                priority_videos.append((filepath, video_dt))
            else:
                other_videos.append((filepath, video_dt))

        # Sort each group by datetime
        priority_videos.sort(key=lambda x: x[1])
        other_videos.sort(key=lambda x: x[1])

        self.stdout.write(f'  Priority videos: {len(priority_videos)}')
        self.stdout.write(f'  Other videos: {len(other_videos)}')

        # Return based on --only flag
        if only_matching:
            return [fp for fp, dt in priority_videos]
        else:
            return [fp for fp, dt in priority_videos] + [fp for fp, dt in other_videos]

    def parse_filename(self, filename):
        """Parse ch<NN>_<YYYYMMDD><HHMMSS>.mp4 filename pattern."""
        match = FILENAME_PATTERN.match(filename)
        if match:
            return (match.group(1), match.group(2), match.group(3))
        return None

    def parse_datetime(self, date_str, time_str, tz):
        """Parse date and time strings into timezone-aware datetime."""
        dt = datetime.strptime(f'{date_str}{time_str}', '%Y%m%d%H%M%S')
        return dt.replace(tzinfo=tz)

    def find_videos(self, directory, pattern, recursive):
        """Find all video files matching the pattern."""
        video_files = []

        # Handle brace expansion patterns like *.{mp4,avi}
        if '{' in pattern and '}' in pattern:
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
