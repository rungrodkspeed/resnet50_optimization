
# Resnet Optimization

While the major focus of this repository is to demonstrate the performance of optimization through the use of many stacks or frameworks, it also implements the Resnet CNN (Convolution Neutal Network) architecture for the classicalation of five flowers.
## Installation

Clone repository

```bash
  git clone https://github.com/rungrodkspeed/resnet50_optimization
```

Install with pip

```bash
  pip install -r requirement.txt
```
For detail TensorRT backed Installation : https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

or Install with docker

```bash
  docker build -t resnet_optim .
```
then
```bash
  docker run --gpus=1 -it --rm resnet_optim
```

**Be careful the image size is 19.5 GiB

## Optimizations

    - smaller size
    - faster
    - more efficient

|Framework | Size (MiB)  |
| ------------- | ------------- |
| Pytorch |  195  |
| ONNX |  97.4  |
| TensorRT |  51.1 |

|Batch(es) | Pytorch (CPU)  | Pytorch (GPU) | onnxruntime (CPU)  | onnxruntime (GPU) | TensorRT |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1 |   17.94 FPS  | 5.79 FPS  | 55.62 FPS  | not supported for CUDA 12  | 395.39 FPS  |
| 8 |   16.82 FPS  | 18.18 FPS  | 47.54 FPS  | not supported for CUDA 12    | 1958.92 FPS |
| 16 |   16.45 FPS  | 72.59 FPS  | 36.69 FPS  | not supported for CUDA 12  | 2154.45 FPS  |
| 32 | 16.26 FPS | 115.96 FPS | 38.22 FPS | not supported for CUDA 12 | 2335.60 FPS |
| 64 |   14.43 FPS  | CUDA out of memory  | 38.05 FPS  | not supported for CUDA 12  | 2523.06 FPS  |

Hardware :
AMD Ryzen 7 5800H with Radeon Graphics 3.20 GHz Processor,
NVIDIA GeForce RTX 3060 Laptop GPU\
nvidia-driver : 531.97\
CUDA version : 12.1

## Deployment

more efficient about performance by Triton stack.

1. Pull images NGC triton inference server.
    ```
    nvcr.io/nvidia/tritonserver:23.06-py3
    ```


2. Create model repository.

    ```
    <model-repository-path>/
        <model-name>/
            config.pbtxt
            1/
                model.plan
    ```

For more detail : https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md

3. Create config.pbtxt
```
name: "resnet50"
platform: "tensorrt_plan"
max_batch_size: 128
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }

]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

default_model_filename: "resnet50.plan"
```

4. Launch Triton Server
    ```
    docker run --name=resnet-triton-container --shm-size='1g' -d --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 resnet-triton
    ```

5. Launch Client
    - Python client
        ```
        docker run --name=resnet-client-python-container -d --rm -p 8888:8888 resnet-client-python
        ```
    - Golang client
        ```
        docker run --name=resnet-client-python-container -d --rm -p 8888:8888 resnet-client-golang
        ```

6. Inference by send requests to Triton server. 
    - Python client
        ```
        python3 /app_python/request.py
        ```
    - Golang client
        ```
        python3 /app_golang/request.py
        ```
## Model Analyzer

Summary about Resnet50 on Triton server. \
(For watch more high resolution at /analyzer_result/reports/summaries/resnet50 directory)
<image src="/analyzer_result/reports/summaries/resnet50/pdf_to_img/result_summary-images-1.jpg"/>
<image src="/analyzer_result/reports/summaries/resnet50/pdf_to_img/result_summary-images-2.jpg"/>

From model-analyzer the best config.pbtxt is :

```
name: "resnet50_config_30"
platform: "tensorrt_plan"
max_batch_size: 64
input {
  name: "input"
  data_type: TYPE_FP32
  dims: 3
  dims: 224
  dims: 224
}
output {
  name: "output"
  data_type: TYPE_FP32
  dims: 1000
}
instance_group {
  count: 4
  kind: KIND_GPU
}
default_model_filename: "resnet50.plan"
dynamic_batching {
}
```

For deep detail : \
(For watch more high resolution at /analyzer_result/reports/detailed/resnet50_config_30 directory)
<image src="analyzer_result/reports/detailed/resnet50_config_30/pdf_to_img/detailed_report-images-1.jpg"/>
<image src="analyzer_result/reports/detailed/resnet50_config_30/pdf_to_img/detailed_report-images-2.jpg"/>
