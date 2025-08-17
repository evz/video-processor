FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
                          make \
                          cmake \
                          python3-dev \
                          python3-pip \
                          python3-setuptools \
                          libcurl4-openssl-dev \
                          libssl-dev \
                          libpq-dev \
                          ffmpeg \
                          libavcodec-dev \
                          libavfilter-dev \
                          libavformat-dev \
                          libavutil-dev \
                          libbz2-dev
RUN mkdir /code
WORKDIR /code

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

ENV NVIDIA_DRIVER_CAPABILITIES=all

# Copy and install Python dependencies first (cached unless requirements.txt changes)
COPY requirements.txt /code/
RUN python3 -m pip config set global.break-system-packages true
RUN pip install -r requirements.txt

# Copy application files last (invalidated when code changes, but deps stay cached)
COPY docker-entrypoint.sh /code/
RUN chmod a+x /code/docker-entrypoint.sh
COPY manage.py /code/
COPY video_processor /code/video_processor
COPY processor /code/processor

EXPOSE 8000
ENTRYPOINT ["/code/docker-entrypoint.sh"]
