from flask import Flask, jsonify, render_template
from src import get_contour, generator
import torch

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True, port=8080)
