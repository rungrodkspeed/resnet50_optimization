import cv2
import numpy as np
import tritonclient.grpc as tritongrpcclient

triton_client = tritongrpcclient.InferenceServerClient(url="localhost:8001")

#docker run --name=resnet-triton-ver1 -d --shm-size='1g' --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 resnet-triton:version1

inputs = []
outputs = []
input_name = "INPUT"
output_name = "OUTPUT"
image_data = cv2.imread('../sample/daisy.jpg')
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
image_data = np.expand_dims(image_data, axis=0)

inputs.append(tritongrpcclient.InferInput(input_name, image_data.shape, "UINT8"))
outputs.append(tritongrpcclient.InferRequestedOutput(output_name, class_count=1000))

inputs[0].set_data_from_numpy(image_data)
results = triton_client.infer(
    model_name="ensemble_resnet50", inputs=inputs, outputs=outputs
)

output0_data = results.as_numpy(output_name)
maxs = np.argmax(output0_data, axis=1)
pred_class = output0_data[0][maxs[0]].decode().split(":")[-1]
print("Result is class: {}".format(pred_class))