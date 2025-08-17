import os
from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from django.conf import settings

from processor.models import Video, VideoChunk, Frame, Detection
from processor.tasks import chunk_video, extract_frames, detect, draw_detections, make_detection_video


class Command(BaseCommand):
    help = 'Process a video synchronously through the complete pipeline'

    def add_arguments(self, parser):
        parser.add_argument('video_path', type=str, help='Path to the video file to process')
        parser.add_argument(
            '--name',
            type=str,
            help='Custom name for the video (defaults to filename)'
        )

    def handle(self, *args, **options):
        video_path = options['video_path']
        
        if not os.path.exists(video_path):
            raise CommandError(f'Video file "{video_path}" does not exist.')
        
        video_name = options.get('name') or os.path.basename(video_path)
        
        self.stdout.write(self.style.SUCCESS(f'📁 Creating video record for: {video_name}'))
        
        with open(video_path, 'rb') as f:
            video = Video.objects.create(video_file=File(f, video_name))
        
        self.stdout.write(self.style.SUCCESS(f'✓ Created video with ID: {video.id}'))
        
        try:
            # Step 1: Chunk the video
            self.stdout.write('⚡ Running chunk_video synchronously...')
            chunk_video(video.id)
            
            # Step 2: Extract frames from all chunks
            self.stdout.write('⚡ Processing video chunks...')
            chunks = VideoChunk.objects.filter(video=video)
            for chunk in chunks:
                self.stdout.write(f'  📋 Processing chunk {chunk.sequence_number}...')
                extract_frames(chunk.id)
            
            # Step 3: Run AI detection on all frames
            self.stdout.write('🔍 Running AI detection on frames...')
            frames = Frame.objects.filter(video_chunk__video=video, status='ENQUEUED')
            frame_count = frames.count()
            
            for i, frame in enumerate(frames, 1):
                self.stdout.write(f'  🤖 Detecting objects in frame {i}/{frame_count}...')
                detect(frame.id)
            
            # Step 4: Draw detection boxes
            self.stdout.write('🎨 Drawing detection boxes...')
            draw_detections(video.id)
            
            # Step 5: Create final output video
            self.stdout.write('🎬 Creating final output video...')
            make_detection_video(video.id)
            
            # Report results
            video.refresh_from_db()
            total_frames = Frame.objects.filter(video_chunk__video=video).count()
            detection_count = Detection.objects.filter(frame__video_chunk__video=video).count()
            
            self.stdout.write(self.style.SUCCESS('✅ Demo processing complete!'))
            self.stdout.write(f'🎯 Video processed with {total_frames} frames')
            self.stdout.write(f'🔍 Found {detection_count} detections')
            
            if detection_count == 0:
                self.stdout.write('')
                self.stdout.write(self.style.WARNING('📭 No animals, people, or vehicles detected in this video'))
                self.stdout.write('This could mean:')
                self.stdout.write('  • The video doesn\'t contain detectable objects')
                self.stdout.write('  • Objects are too small/blurry for the AI to detect')
                self.stdout.write('  • Try a different video with clearer wildlife/people/vehicles')
            elif video.detections_file:
                self.stdout.write(f'📺 Output video available at: {video.detections_file.name}')
                
                # Convert container path to host path
                container_path = video.detections_file.path
                # Remove the /output prefix (container mount point) to get relative path
                if container_path.startswith('/output/'):
                    host_relative_path = container_path[8:]  # Remove '/output/'
                else:
                    host_relative_path = container_path
                
                self.stdout.write(f'📁 Host path: {host_relative_path}')
                self.stdout.write('')
                self.stdout.write('🎥 To watch the video:')
                self.stdout.write(f'   # macOS:   open "{host_relative_path}"')
                self.stdout.write(f'   # Windows: start "" "{host_relative_path}"')
                self.stdout.write(f'   # Linux:   xdg-open "{host_relative_path}"')
                self.stdout.write(f'   # Or open {host_relative_path} with your preferred video player')
            else:
                self.stdout.write('')
                self.stdout.write(self.style.WARNING('⚠️  Detections were found but no output video was generated'))
                self.stdout.write('This might indicate a processing error during video creation.')
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Processing failed: {str(e)}'))
            raise CommandError(f'Video processing failed: {str(e)}')