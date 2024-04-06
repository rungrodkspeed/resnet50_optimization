FROM nvcr.io/nvidia/tritonserver:23.06-py3

RUN apt update && \
    apt install -y libopencv-dev

WORKDIR /

COPY ./models /models

RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python

ENV CUDA_MODULE_LOADING=LAZY

CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]