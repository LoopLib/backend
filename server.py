from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

# Import CORS from flask_cors
app = Flask(__name__)
# Add CORS to the app
CORS(app)
# Set the upload folder
app.config['UPLOAD_FOLDER'] = './uploads'
# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)