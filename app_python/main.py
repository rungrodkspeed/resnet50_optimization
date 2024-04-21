import io
import os
import time
import datetime
import traceback
import numpy as np
import tritonclient.grpc as grpcclient

from PIL import Image
from dotenv import load_dotenv
from tritonclient.utils import *
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

load_dotenv(dotenv_path="../.env")

tritonclient = grpcclient.InferenceServerClient(f"{os.environ['TRITON_SERVER_URL']}")
app = FastAPI()

@app.get("/")
def healthcheck() -> bool:
    return True

@app.post("/classify")
async def inference(ImageByte: UploadFile = File(...)):
    
    try:
        image = Image.open( io.BytesIO(ImageByte.file.read()) )
        
    except Exception as e:
        
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    ext = image.format
    
    if ext not in ["JPEG", "PNG", "GIF", "BMP", "TIFF"]:
        return JSONResponse(content={"error": "Image extension not supported."}, status_code=500)
    
    timestamp = int(time.time())
    dt_utc = datetime.datetime.utcfromtimestamp(timestamp)
    gmt7_offset = datetime.timedelta(hours=7)
    dt_gmt7 = dt_utc + gmt7_offset
    formatted_date = dt_gmt7.strftime("%d/%m/%Y %H:%M:%S")
    
    logs = {
        "datetime": formatted_date,
        "shape": image.size,
        "extension": ext
    }
    
    print(logs)
    
    
    try:
        
        image_np = np.asarray( image )
        image_np = np.expand_dims(image_np, axis=0)
        
        print(image_np.shape)

        inputs = [ grpcclient.InferInput("INPUT", image_np.shape, np_to_triton_dtype(image_np.dtype)) ]
        outputs = [ grpcclient.InferRequestedOutput("OUTPUT", class_count=1000) ]

        inputs[0].set_data_from_numpy(image_np)
        
        results = tritonclient.infer(
            model_name="ensemble_resnet50", inputs=inputs, outputs=outputs
        )

        output0_data = results.as_numpy("OUTPUT")
        results = output0_data[0][0].decode().split(":")
        
        print(output0_data)
        
        confident = float(results[0])
        class_name = results[-1]
        
        return JSONResponse(content={"class": class_name, "confident":confident}, status_code=200)
        
    except Exception as e:
        
        return JSONResponse(content={"error": str(e)}, status_code=500)
    