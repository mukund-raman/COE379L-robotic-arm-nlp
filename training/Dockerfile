# Image: mkr2497/robotic-arm-nlp:latest
# Build using: docker build -t mkr2497/robotic-arm-nlp:latest .
# Run using: docker run -it --rm -p 5000:5000 mkr2497/robotic-arm-nlp:latest
FROM python:3.12

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip
RUN pip install flask ultralytics opencv-python numpy

COPY ./runs/detect/train/weights/best.pt /yolov11-candy.pt
COPY ./runs/detect/train2/weights/best.pt /yolov11-bottle.pt
COPY ./runs/detect/train3/weights/best.pt /yolov11-coin.pt
COPY api.py /api.py

CMD ["python", "api.py"]