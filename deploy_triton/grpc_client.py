import numpy as np
import tritonclient.grpc as grpcclient

def test_infer(model_name, data):
    triton_client = grpcclient.InferenceServerClient(url='localhost:8001')

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input', list(data.shape), "FP32"))

    inputs[0].set_data_from_numpy(data)

    outputs.append(grpcclient.InferRequestedOutput('output'))

    results = triton_client.infer(
        model_name,
        inputs,
        model_version="1",
        outputs=outputs)

    return results

if __name__ == '__main__':
    
    data = np.random.rand(8, 3, 224, 224).astype(np.float32)

    res = test_infer('resnet50', data)

    print(res)
