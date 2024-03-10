# Generated by Django 5.0.2 on 2024-03-10 14:31

import processor.models
import storages.backends.s3
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0007_remove_frame_frame_data_remove_video_path_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='frame',
            name='frame_file',
            field=models.FileField(null=True, storage=storages.backends.s3.S3Storage(bucket_name='evz-frames-bucket'), upload_to=processor.models.frame_upload_path),
        ),
    ]
