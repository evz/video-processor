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
                          libbz2-dev \
                          git \
                          wget \
                          unzip
RUN mkdir /code
WORKDIR /code

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

ENV NVIDIA_DRIVER_CAPABILITIES=all

# Copy and install Python dependencies first (cached unless requirements.txt changes)
COPY requirements.txt /code/
RUN python3 -m pip config set global.break-system-packages true
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy and install NVIDIA Video Codec SDK (any version)
# The build expects a file matching Video_Codec_SDK_*.zip in the project root
COPY Video_Codec_SDK_*.zip /tmp/video_codec_sdk.zip
RUN unzip -q /tmp/video_codec_sdk.zip -d /tmp && \
    SDK_DIR=$(find /tmp -maxdepth 1 -name "Video_Codec_SDK_*" -type d | head -1) && \
    cp -r ${SDK_DIR}/Interface/* /usr/local/include/ && \
    mkdir -p /usr/local/cuda/lib64/stubs && \
    cp ${SDK_DIR}/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/stubs/ && \
    rm -rf /tmp/video_codec_sdk.zip ${SDK_DIR}

ARG CACHE_BUST=1
# Build and install decord from fork with CUDA support
# Temporarily add stubs to library path for build, then remove them
RUN git clone --recursive https://github.com/evz/decord /tmp/decord && \
    cd /tmp/decord && \
    mkdir build && \
    cd build && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH make -j$(nproc) && \
    cd /tmp/decord/python && \
    python3 setup.py install && \
    rm -rf /tmp/decord && \
    # Remove stub libraries from stubs directory to prevent runtime conflicts
    rm -f /usr/local/cuda/lib64/stubs/libnvcuvid.so /usr/local/cuda/lib64/stubs/libnvidia-encode.so && \
    ldconfig

# Copy MegaDetector model to avoid re-downloading on every container start
COPY models/md_v5a.0.0.pt /models/md_v5a.0.0.pt

ENV MDV5A=/models/md_v5a.0.0.pt

# Copy application files last (invalidated when code changes, but deps stay cached)
COPY docker-entrypoint.sh /code/
RUN chmod a+x /code/docker-entrypoint.sh
COPY manage.py /code/
COPY video_processor /code/video_processor
COPY processor /code/processor

EXPOSE 8000
ENTRYPOINT ["/code/docker-entrypoint.sh"]
