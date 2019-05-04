import PIL.Image
import numpy
import requests
from pprint import pprint
import time

image = PIL.Image.open("moose_test1.jpeg")
image_np = numpy.array(image)


payload = {"instances": [image_np.tolist()]}
start = time.perf_counter()
res = requests.post(
    "http://10.100.6.187:8501/v1/models/ssdlite_mobilenet_v2_coco_2018_05_09:predict", json=payload)
print(f"Took {time.perf_counter()-start:.2f}s")
pprint(res.json())
