FROM pytorch/pytorch

WORKDIR /

COPY models/ggml-large.bin /models/ggml-large.bin

RUN apt update && apt install -y libsndfile1 ffmpeg git build-essential

RUN git clone https://github.com/ggerganov/whisper.cpp.git && cd /whisper.cpp && make && chmod +x ./main

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp