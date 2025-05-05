from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
model = YOLO('yolov11.pt')  # Load the YOLOv11 model

@app.route('/models/yolov11/bottles/v1', methods=['GET'])
def model_info():
    # Get the model details
    num_params = sum(p.numel() for p in model.model.parameters())
    return {
        "version": "v1",
        "name": "water-bottle-detector",
        "description": "Detects locations of water bottles in images.",
        "number_of_parameters": num_params,
        "architecture": model.yaml
    }

@app.route('/models/yolov11/bottles/v1', methods=['POST'])
def predict():
    # Get the image from the request and perform inference
    image_file = request.files['image']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    results = model.predict(source=image, save=False)
    
    # Draw bounding boxes on the image
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"Class {int(cls)}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, image)
    temp_file.close()
    
    # Return the image file
    return send_file(temp_file.name, mimetype='image/jpeg')

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')