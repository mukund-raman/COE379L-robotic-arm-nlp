from flask import Flask, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

app = Flask(__name__)
bottle_model = YOLO('yolov11-bottle.pt')
candy_model = YOLO('yolov11-candy.pt')
coin_model = YOLO('yolov11-coin.pt')

def model_info(model, name, object_detector):
    # Get the model details
    num_params = sum(p.numel() for p in model.model.parameters())
    return {
        "version": "v1",
        "name": name,
        "description": f"Detects locations of {object_detector} in images.",
        "number_of_parameters": num_params,
        "architecture": model.yaml
    }

def predict(model):
    # Get the image from the request and perform inference
    image_file = request.files['image']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    results = model.predict(source=image, save=False)
    
    # Get model's class names and draw bounding boxes on the image
    class_names = model.names if hasattr(model, 'names') else {}
    for result in results:
        boxes = result.boxes.xyxy.numpy()
        scores = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names.get(int(cls), f"Class {int(cls)}")
            label = f"{class_name}: {score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, image)
    temp_file.close()
    
    # Return the image file
    return send_file(temp_file.name, mimetype='image/jpeg')


"""YOLOv11 Water Bottle Detection Model"""

@app.route('/models/yolov11/bottles/v1', methods=['GET'])
def bottle_model_info():
    return model_info(bottle_model, "water-bottle-detector", "water bottles")

@app.route('/models/yolov11/bottles/v1', methods=['POST'])
def bottle_predict():
    return predict(bottle_model)


"""YOLOv11 Candy Detection Model"""

@app.route('/models/yolov11/candy/v1', methods=['GET'])
def candy_model_info():
    return model_info(candy_model, "candy-detector", "candy")
    
@app.route('/models/yolov11/candy/v1', methods=['POST'])
def candy_predict():
    return predict(candy_model)


"""YOLOv11 Coin Detection Model"""

@app.route('/models/yolov11/coins/v1', methods=['GET'])
def coin_model_info():
    return model_info(coin_model, "coin-detector", "coins")

@app.route('/models/yolov11/coins/v1', methods=['POST'])
def coin_predict():
    return predict(coin_model)

# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')