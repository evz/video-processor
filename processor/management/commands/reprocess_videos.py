"""
Management command to reprocess stuck videos.

Usage:
    python manage.py reprocess_videos --stuck-chunks
    python manage.py reprocess_videos --ids 6000 6001 6002
    python manage.py reprocess_videos --range 6000 6064
    python manage.py reprocess_videos --fix-completed
"""

from celery import chain
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Sum
from processor.models import Video, VideoChunk, Frame
from processor.tasks import chunk_video, extract_frames, draw_detections, track_and_filter_detections


class Command(BaseCommand):
    help = 'Reprocess stuck videos that failed during worker restarts'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stuck-chunks',
            action='store_true',
            help='Requeue videos that have status=PROCESSING but no chunks created'
        )
        parser.add_argument(
            '--ids',
            nargs='+',
            type=int,
            help='Specific video IDs to reprocess'
        )
        parser.add_argument(
            '--range',
            nargs=2,
            type=int,
            metavar=('START', 'END'),
            help='Range of video IDs to reprocess (inclusive)'
        )
        parser.add_argument(
            '--fix-completed',
            action='store_true',
            help='Fix videos that have all frames completed but are still marked PROCESSING'
        )
        parser.add_argument(
            '--resume-stuck',
            action='store_true',
            help='Resume videos with chunks stuck in PROCESSING (interrupted extraction)'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be reprocessed without actually doing it'
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']

        if dry_run:
            self.stdout.write(self.style.WARNING('\nDRY RUN - No changes will be made\n'))

        if options['fix_completed']:
            self.fix_completed_videos(dry_run)

        if options['stuck_chunks']:
            self.reprocess_stuck_chunks(dry_run)

        if options['resume_stuck']:
            self.resume_stuck_extraction(dry_run)

        if options['ids']:
            self.reprocess_by_ids(options['ids'], dry_run)

        if options['range']:
            start_id, end_id = options['range']
            video_ids = list(range(start_id, end_id + 1))
            self.reprocess_by_ids(video_ids, dry_run)

        if not any([options['stuck_chunks'], options['ids'], options['range'],
                     options['fix_completed'], options['resume_stuck']]):
            self.stdout.write(self.style.WARNING(
                'No action specified. Use --stuck-chunks, --resume-stuck, --ids, --range, or --fix-completed'
            ))

    def fix_completed_videos(self, dry_run):
        """Fix videos that have all chunks completed but are still marked PROCESSING."""
        self.stdout.write('\n=== Fixing videos with all chunks completed ===\n')

        fixed_count = 0
        for video in Video.objects.filter(status='PROCESSING'):
            chunks = VideoChunk.objects.filter(video=video)

            # Skip if no chunks yet
            if not chunks.exists():
                continue

            # Skip if any chunks are incomplete
            if chunks.exclude(status='COMPLETED').exists():
                continue

            # Use sum of chunk frame counts, not original video.frame_count
            # (FFmpeg segmentation can lose frames at boundaries)
            chunk_frame_total = chunks.aggregate(total=Sum('frame_count'))['total'] or 0
            completed_frames = Frame.objects.filter(
                video_chunk__video_id=video.id,
                status='COMPLETED'
            ).count()

            if completed_frames >= chunk_frame_total:
                self.stdout.write(
                    f'Video {video.id} ({video.name}): {completed_frames}/{chunk_frame_total} frames completed '
                    f'(original video had {video.frame_count})'
                )

                if not dry_run:
                    video.status = 'COMPLETED'
                    video.save()
                    # Chain tracking and drawing tasks
                    if settings.USE_TRACKING:
                        chain(
                            track_and_filter_detections.s(video.id),
                            draw_detections.s()
                        ).delay()
                    else:
                        draw_detections.delay(video.id)
                    self.stdout.write(self.style.SUCCESS(f'  -> Marked as COMPLETED and queued for output generation'))
                else:
                    self.stdout.write(self.style.WARNING(f'  -> Would mark as COMPLETED'))
                fixed_count += 1

        self.stdout.write(f'\nFixed {fixed_count} video(s)\n')

    def resume_stuck_extraction(self, dry_run):
        """Resume videos with chunks stuck in PROCESSING status."""
        self.stdout.write('\n=== Resuming stuck PROCESSING chunks ===\n')

        stuck_chunks = VideoChunk.objects.filter(status='PROCESSING')
        affected_videos = Video.objects.filter(
            id__in=stuck_chunks.values_list('video_id', flat=True).distinct()
        )

        self.stdout.write(
            f'Found {stuck_chunks.count()} stuck chunk(s) '
            f'across {affected_videos.count()} video(s)\n'
        )

        requeued = 0
        for video in affected_videos.order_by('id'):
            video_stuck = stuck_chunks.filter(video=video)
            total_chunks = VideoChunk.objects.filter(video=video).count()
            completed_chunks = VideoChunk.objects.filter(video=video, status='COMPLETED').count()

            total_frames = Frame.objects.filter(video_chunk__video=video).count()
            expected = video.frame_count or 0

            self.stdout.write(
                f'Video {video.id} ({video.name}): '
                f'{video_stuck.count()} stuck chunks, '
                f'{completed_chunks}/{total_chunks} completed, '
                f'{total_frames}/{expected} frames'
            )

            if not dry_run:
                for chunk in video_stuck:
                    extract_frames.delay(chunk.id)
                self.stdout.write(self.style.SUCCESS(
                    f'  -> Requeued {video_stuck.count()} chunk(s) for extraction'
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f'  -> Would requeue {video_stuck.count()} chunk(s)'
                ))
            requeued += video_stuck.count()

        self.stdout.write(f'\nRequeued {requeued} chunk(s)\n')

    def reprocess_stuck_chunks(self, dry_run):
        """Requeue videos that have status=PROCESSING but no chunks created."""
        self.stdout.write('\n=== Reprocessing videos stuck at chunking ===\n')

        stuck_videos = Video.objects.filter(
            status='PROCESSING'
        ).exclude(
            id__in=VideoChunk.objects.values_list('video_id', flat=True)
        )

        self.stdout.write(f'Found {stuck_videos.count()} video(s) with no chunks\n')

        requeued = 0
        for video in stuck_videos:
            # Check if source file exists
            if not video.video_file or not video.video_file.storage.exists(video.video_file.name):
                self.stdout.write(self.style.ERROR(
                    f'Video {video.id} ({video.name}): Source file missing, cannot reprocess'
                ))
                continue

            self.stdout.write(f'Video {video.id} ({video.name})')

            if not dry_run:
                # Reset status to ENQUEUED so chunk_video will process it
                video.status = 'ENQUEUED'
                video.save()
                chunk_video.delay(video.id)
                self.stdout.write(self.style.SUCCESS(f'  -> Requeued for chunking'))
            else:
                self.stdout.write(self.style.WARNING(f'  -> Would requeue for chunking'))
            requeued += 1

        self.stdout.write(f'\nRequeued {requeued} video(s)\n')

    def reprocess_by_ids(self, video_ids, dry_run):
        """Reprocess specific video IDs."""
        self.stdout.write(f'\n=== Reprocessing {len(video_ids)} video(s) by ID ===\n')

        requeued = 0
        for video_id in video_ids:
            try:
                video = Video.objects.get(id=video_id)
            except Video.DoesNotExist:
                self.stdout.write(self.style.ERROR(f'Video {video_id}: Not found'))
                continue

            # Check current state
            chunk_count = VideoChunk.objects.filter(video=video).count()
            frame_count = Frame.objects.filter(video_chunk__video=video).count()

            self.stdout.write(
                f'Video {video.id} ({video.name}): status={video.status}, '
                f'chunks={chunk_count}, frames={frame_count}'
            )

            if not dry_run:
                if chunk_count == 0:
                    # Need to start from chunking
                    video.status = 'ENQUEUED'
                    video.save()
                    chunk_video.delay(video.id)
                    self.stdout.write(self.style.SUCCESS(f'  -> Requeued for chunking'))
                elif frame_count < (video.frame_count or 0):
                    # Have chunks but missing frames - requeue incomplete chunks
                    incomplete_chunks = VideoChunk.objects.filter(
                        video=video, status__in=['ENQUEUED', 'PROCESSING']
                    )
                    for chunk in incomplete_chunks:
                        extract_frames.delay(chunk.id)
                    self.stdout.write(self.style.SUCCESS(
                        f'  -> Requeued {incomplete_chunks.count()} incomplete chunks for extraction'
                    ))
                else:
                    self.stdout.write(self.style.WARNING(f'  -> Already has all frames'))
            else:
                self.stdout.write(self.style.WARNING(f'  -> Would requeue'))
            requeued += 1

        self.stdout.write(f'\nProcessed {requeued} video(s)\n')
