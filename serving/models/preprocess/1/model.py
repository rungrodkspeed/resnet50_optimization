import cv2
import json
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        
        self.model_config = model_config = json.loads(args["model_config"])
        
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_0")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )
        
        self.mean = np.array( [0.485, 0.456, 0.406] ).reshape((3, 1, 1))
        self.std = np.array( [0.229, 0.224, 0.225] ).reshape((3, 1, 1))

    def transforms(self, image):
        resized_image = cv2.resize(image, (256, 256))
        cropped_image = resized_image[16:240, 16:240]
        tensor_image = cropped_image.transpose((2, 0, 1))
        normalized_image = ( ( tensor_image / 255.0 ) -  self.mean) / self.std
        return normalized_image.astype(np.float32)

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            img = in_0.as_numpy()

            processed_img = list( map (lambda x: self.transforms(x), img) )

            out_tensor_0 = pb_utils.Tensor( "OUTPUT_0", np.asarray( processed_img) )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Preprocess Cleaning up...")