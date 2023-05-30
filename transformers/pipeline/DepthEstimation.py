from transformers import pipeline
checkpoint = "vinvino02/glpn-nyu"
depth_estimator = pipeline("depth-estimation", model=checkpoint, device=1)

from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg"
image = Image.open(requests.get(url, stream=True).raw)
predictions = depth_estimator(image)

print(predictions)
