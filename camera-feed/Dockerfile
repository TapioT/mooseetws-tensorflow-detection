# if running on rapberry pi
#FROM arm32v7/python:3
# if running on x86 based PC
FROM python:3

RUN pip install opencv-python

COPY start.py /home/demo/start.py

CMD ["python", "/home/demo/start.py"]