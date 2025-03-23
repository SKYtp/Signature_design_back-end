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

### install (Options)
pip install torch torchvision torchaudio
pip install ultralytics
pip install quart-cors
pip install Flask
pip install Pillow
pip install aiomysql
pip install python-dotenv
pip install hypercorn

### run
hypercorn app:app --bind 0.0.0.0:8080