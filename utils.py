import os
import glob
import numpy as np
import tensorrt as trt

EPSILON = 1.1102230246251565e-16

classes = {0:'daisy', 1:'dandelion', 2:'rose', 3:'sunflower', 4:'tulip'}

dtype_np={'INT8':np.int8,
           'HALF':np.half,
           'FLOAT':np.float32,
           'DOUBLE':np.float64,}

dtype_trt={'INT8':trt.BuilderFlag.INT8,
            'HALF':trt.BuilderFlag.FP8,
            'FLOAT':trt.BuilderFlag.FP16,
            'DOUBLE':trt.BuilderFlag.TF32,}

def get_path(directory_path):
    ext = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp', '*.svg']
    image_paths = []
    for extension in ext:
        image_paths.extend(glob.glob(os.path.join(directory_path, '**', extension), recursive=True))

    return image_paths