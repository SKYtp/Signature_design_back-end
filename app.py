from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from src import get_contour, generator, option_to_meaning, value_to_point, connect, contrast
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
    
    # image_path = os.path.join(os.getcwd(), "public/images/final_image.png")

    text = option_to_meaning.option_2_meaning(received_data)

    sig_name = received_data.get("name")
    
    if(received_data.get("boss") == "edge"):
        sig_style = 1
    else:
        sig_style = 0

    if(received_data.get("symbol") == "omega"):
        sig_symbol = 1
    elif(received_data.get("symbol") == "loop"):
        sig_symbol = 2
    else:
        sig_symbol = 0
    
    if(received_data.get("tilt") == "tilt_up"):
        sig_tilt = True
    else:
        sig_tilt = False
    
    if(received_data.get("point") == "checked_point"):
        sig_dot = True
    else:
        sig_dot = False

    if(received_data.get("line") == "checked_line"):
        sig_line = True
    else:
        sig_line = False

    sig_data = connect.v_concat(sig_name, sig_style, sig_symbol, sig_tilt, sig_dot, sig_line)

    print("angle: ",sig_data.get("angle")," tall_ratio: ",sig_data.get("tall_ratio")," distance: ",sig_data.get("distance"), " head_broken: ",sig_data.get("head_broken"), " head_cross: ",sig_data.get("head_cross"))

    points = {
        "point1": value_to_point.value_2_point1(sig_data.get("angle")), # ตำแหน่งประธานต้องอยู่ในระนาบเดียวกับตำแหน่งบริวาร
        "point2": value_to_point.value_2_point2_3(sig_data.get("tall_ratio")), # ความสูงบริวารต้องเป็นเศษหนึ่งส่วนสองของความสูงประธาน
        "point3": value_to_point.value_2_point2_3(sig_data.get("distance")), # ประธานกับบริวารต้องเว้นว่างเป็นเศษหนึ่งส่วนสองของความสูงบริวาร
        "point4": value_to_point.value_2_point4(sig_data.get("head_broken")), # ตัวอักษรในลายเซ็นจะต้องไม่มีการขาดของเส้นภายในตัวอักษร
        "point5": value_to_point.value_2_point5(sig_data.get("head_cross"), sig_data.get("head_is")) # ประธานต้องไม่มีเส้นตัดกันที่เกิดจากการเซ็น
    }

    print(points)
    # points = json.dumps(points)
    
    try:
        # with open(image_path, "rb") as image_file:
        #     base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
        base64_image = contrast.increase_contrast(sig_data.get('image'), 1.7)
        base64_image = f"data:image/png;base64,{base64_image}"
        
        return jsonify({
            "message": "Success",
            "image": base64_image,
            "text": text,
            "points": points
        })
    except Exception as e:
        return jsonify({"error": "Error reading image", "details": str(e)}), 500
    
@app.route('/inquiry', methods=['POST'])
def get_inquiry():
    received_data = request.get_json()
    # print("Received JSON data:", received_data)

    return jsonify({
        "message": "Success",
        "receivedData": received_data,
    })



if __name__ == '__main__':

    try:
        app.run(debug=True, port=8080)
    except Exception as e:
        print(f"App crashed: {e}")