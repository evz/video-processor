FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

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

COPY decord /code/decord
COPY decord/Video_Codec_SDK_12.2.72/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/stubs/
RUN echo "/usr/local/cuda/lib64/stubs\n" > /etc/ld.so.conf.d/999-cuda-stubs.conf && ldconfig
RUN cd decord && rm -rf build && mkdir build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_Build_Type=Release && make && cd ../python && python3 setup.py install

RUN pip install --upgrade pip
COPY requirements.txt /code/
COPY docker-entrypoint.sh /code/
RUN chmod a+x /code/docker-entrypoint.sh
COPY manage.py /code/
COPY video_processor /code/video_processor
COPY processor /code/processor

RUN pip install --upgrade "numpy<2.0"
RUN pip install -r requirements.txt

EXPOSE 8000
ENTRYPOINT ["/code/docker-entrypoint.sh"]
