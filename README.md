# Signature_design_back-end
## This is flask for Signature design project

### For create python venv
python -m venv .venv

### For activate an env
.venv\Scripts\activate

### For install dependencies for this project
pip install -r requirements.txt

### For update dependencies
pip freeze > requirements.txt

### pytorch
pip install torch torchvision torchaudio
pip install ultralytics
pip install quart-cors

pip install Flask
pip install Pillow
pip install aiomysql


flask run --host=0.0.0.0 --port=8080