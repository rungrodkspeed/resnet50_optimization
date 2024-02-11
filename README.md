
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

or Install with docker

```bash
  docker
```

For detail TensorRT backed Installation : https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
## Optimizations

    - smaller size
    - faster
    - more efficient

|Framework \ Size | Size (MiB)  |
| ------------- | ------------- |
| Pytorch |  195  |
| ONNX |  97.4  |
| TensorRT |  51.1 |

|Batch(es) \ Framework | Pytorch (CPU)  | Pytorch (GPU) | onnxruntime (CPU)  | onnxruntime (GPU) | TensorRT |
| ------------- | ------------- | ------------- || ------------- | ------------- || ------------- |
| 1 |   17.94 FPS  | 5.79 FPS  || 55.62 FPS  | not supported for CUDA 12  || 395.39 FPS  |
| 8 |   16.82 FPS  | 18.18 FPS  || 47.54 FPS  | not supported for CUDA 12    || 1958.92 FPS |
| 16 |   16.45 FPS  | 72.59 FPS  || 36.69 FPS  | not supported for CUDA 12  || 2154.45 FPS  |
| 32 | 16.26 FPS | 115.96 FPS || 38.22 FPS | not supported for CUDA 12 || 2335.60 FPS |
| 64 |   14.43 FPS  | CUDA out of memory  || 38.05 FPS  | not supported for CUDA 12  || 2523.06 FPS  |

Hardware :
AMD Ryzen 7 5800H with Radeon Graphics 3.20 GHz Processor,
NVIDIA GeForce RTX 3060 Laptop GPU\
nvidia-driver : 531.97\
CUDA version : 12.1

## Deployment

more efficient about performance by Triton stack.

1. Pull images NGC triton inference server.
    ```
    nvcr.io/nvidia/tritonserver:23.07-py3
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

4. Launch NGC for deploy our model.
    ```
    docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -e CUDA_MODULE_LOADING=LAZY -v/your/path/models:/models nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-repository=/models --log-verbose=1
    ```

5. Inference by send requests to Triton server. 
    - HTTP requests
        ```
        python3 /deploy_triton/client.py
        ```
    - gRPC requests
        ```
        python3 /deploy_triton/grpc_client.py
        ```
## Model Analyzer

Summary about Resnet50 on Triton server.

<a href="analyzer_result/reports/summaries/resnet50/result_summary.pdf" class="image fit"><img src="images/marr_pic.jpg" alt=""></a>

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

For deep detail :

<a href="analyzer_result/reports/detailed/resnet50_config_30/detailed_report.pdf" class="image fit"><img src="images/marr_pic.jpg" alt=""></a>