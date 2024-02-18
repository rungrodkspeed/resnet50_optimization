FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

ENV CUDA_MODULE_LOADING=LAZY

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace

COPY . .

RUN ["bash"]