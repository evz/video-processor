# Generated by Django 5.0.2 on 2024-06-29 13:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0015_videochunk_sequence_number'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='frame_rate',
            field=models.IntegerField(default=20),
        ),
    ]
