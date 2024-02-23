FROM python:3.12-bookworm

RUN mkdir /code
WORKDIR /code
RUN apt-get update && apt-get -y install libcurl4-openssl-dev libssl-dev libpq-dev ffmpeg
RUN pip install --upgrade pip
COPY requirements.txt /code/
COPY docker-entrypoint.sh /code/
RUN chmod a+x /code/docker-entrypoint.sh
COPY manage.py /code/
COPY video_processor /code/video_processor
COPY processor /code/processor

RUN pip install -r requirements.txt

EXPOSE 8000
ENTRYPOINT ["/code/docker-entrypoint.sh"]
