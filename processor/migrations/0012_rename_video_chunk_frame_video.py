# Generated by Django 5.0.2 on 2024-06-27 12:09

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0011_alter_frame_video_chunk_delete_videochunk'),
    ]

    operations = [
        migrations.RenameField(
            model_name='frame',
            old_name='video_chunk',
            new_name='video',
        ),
    ]
