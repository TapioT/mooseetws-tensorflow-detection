FROM arm32v7/python:2.7

RUN apt update && apt install -y python-opencv

COPY start.py /home/demo/start.py

EXPOSE 5000

CMD ["python", "/home/demo/start.py"]