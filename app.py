from flask import Flask
from src import get_contour

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/contour')
def test_contour():
    image_path = './test_pic/'
    all_contour = get_contour.extract_all_contour_points(image_path + '21.png')
    return all_contour.tolist()
if __name__ == '__main__':
    app.run(debug=True, port=8080)
