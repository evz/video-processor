# Generated by Django 5.0.2 on 2024-06-16 14:21

import django.core.files.storage
import processor.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0008_alter_frame_frame_file'),
    ]

    operations = [
        migrations.AlterField(
            model_name='frame',
            name='frame_file',
            field=models.FileField(null=True, storage=django.core.files.storage.FileSystemStorage(base_url='frames', location='/home/eric/code/animal-celery/video_processor/frames'), upload_to=processor.models.frame_upload_path),
        ),
        migrations.AlterField(
            model_name='video',
            name='frame_count',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='video',
            name='name',
            field=models.CharField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='video',
            name='status',
            field=models.CharField(choices=[('ENQUEUED', 'Enqueued'), ('PROCESSING', 'Processing'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')], default='ENQUEUED', max_length=10),
        ),
        migrations.AlterField(
            model_name='video',
            name='video_file',
            field=models.FileField(default='', storage=django.core.files.storage.FileSystemStorage(base_url='videos', location='/home/eric/code/animal-celery/video_processor/videos'), upload_to=''),
            preserve_default=False,
        ),
    ]
