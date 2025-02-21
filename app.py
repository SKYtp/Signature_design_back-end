from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from src import get_contour, generator
import torch
import os
import base64
import json

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/contour')
def test_contour():
    image_path = './test_pic/'
    all_contour = get_contour.extract_all_contour_points(image_path + '21.png')
    return all_contour.tolist()

@app.route('/generate_image_starter_curve')
def generate_image_starter_curve():
    model = "ส_starter"
    base64_image = generator.generate(model, torch.tensor([0]))
    return jsonify(image=base64_image, status=200)

@app.route('/generate_image_starter_hard')
def generate_image_starter_hard():
    model = "ส_starter"
    base64_image = generator.generate(model, torch.tensor([1]))
    return jsonify(image=base64_image, status=200)

@app.route('/generate_image_follower')
def generate_image_follower():
    model = "ข_follower_front"
    base64_image = generator.generate(model)
    return jsonify(image=base64_image, status=200)

@app.route('/from-data-to-image', methods=['POST'])
def from_data_to_image():
    received_data = request.get_json()
    print("Received JSON data:", received_data)
    
    image_path = os.path.join(os.getcwd(), "public/images/final_image.png")

    text = """
    Lorem ipsum dolor sit amet, consectetur adipiscing
    elit. Suspendisse molestie euismod interdum. Donec 
    blandit ligula ut leo dictum, non viverra ligula ultrices. 
    Nullam pulvinar iaculis enim, vitae eleifend justo 
    semper ut. Nunc velit magna, feugiat a nulla eget, 
    consequat ultricies nisl. Quisque tortor tellus
    """
    text += "ddddsdd"
    text += "\ntest"

    points = {
        "point1": 1.3,
        "point2": 2.2,
        "point3": 3,
        "point4": 4,
        "point5": 5
    }

    # points = json.dumps(points)
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
        
        return jsonify({
            "message": "Success",
            "receivedData": received_data,
            "image": base64_image,
            "text": text,
            "points": points
        })
    except Exception as e:
        return jsonify({"error": "Error reading image", "details": str(e)}), 500
    
@app.route('/get-inquiry', methods=['POST'])
def get_inquiry():
    received_data = request.get_json()
    # print("Received JSON data:", received_data)

    return jsonify({
        "message": "Success",
        "receivedData": received_data,
    })



if __name__ == '__main__':
    app.run(debug=True, port=8080)
