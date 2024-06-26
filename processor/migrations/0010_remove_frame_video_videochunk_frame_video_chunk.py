# Generated by Django 5.0.2 on 2024-06-27 00:52

import django.core.files.storage
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0009_alter_frame_frame_file_alter_video_frame_count_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='frame',
            name='video',
        ),
        migrations.CreateModel(
            name='VideoChunk',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=500)),
                ('video_file', models.FileField(storage=django.core.files.storage.FileSystemStorage(base_url='videos', location='/home/eric/code/animal-celery/video_processor/videos'), upload_to='')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('video', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='processor.video')),
            ],
        ),
        migrations.AddField(
            model_name='frame',
            name='video_chunk',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='processor.videochunk'),
        ),
    ]
