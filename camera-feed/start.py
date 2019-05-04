import time
from SimpleCV import Camera

cam = Camera()
time.sleep(1)  # If you don't wait, the image will be dark
img = cam.getImage()
img.save("simplecv.png")