import requests

image_path = "../sample/daisy.jpg"

with open(image_path, 'rb') as image_file:
    files = {'image_file': image_file}
    response = requests.post("http://localhost:8888/classify", files=files)

if response.status_code == 200:
    print("Inference successful")
    print(response.text)
    
else:
    print("Inference failed")
    print(response.text)