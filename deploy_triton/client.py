import numpy as np
import tritonclient.http as httpclient

triton_client = httpclient.InferenceServerClient(url='localhost:8000', verbose=True, network_timeout=60.0)

def test_infer(model_name, data):
    
    inputs = []
    outputs = []
    inputs.append(httpclient.InferInput('input', list(data.shape), "FP32"))

    inputs[0].set_data_from_numpy(data)

    outputs.append(httpclient.InferRequestedOutput('output', binary_data=False, class_count=1000))
    
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