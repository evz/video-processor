# Generated by Django 5.0.2 on 2024-02-15 22:14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('processor', '0002_frame_frame_path_detection'),
    ]

    operations = [
        migrations.RenameField(
            model_name='detection',
            old_name='y_cood',
            new_name='y_coord',
        ),
    ]