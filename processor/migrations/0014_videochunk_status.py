# Generated by Django 5.0.2 on 2024-06-28 21:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0013_remove_frame_video_videochunk_frame_video_chunk'),
    ]

    operations = [
        migrations.AddField(
            model_name='videochunk',
            name='status',
            field=models.CharField(choices=[('ENQUEUED', 'Enqueued'), ('PROCESSING', 'Processing'), ('COMPLETED', 'Completed'), ('FAILED', 'Failed')], default='ENQUEUED', max_length=10),
        ),
    ]