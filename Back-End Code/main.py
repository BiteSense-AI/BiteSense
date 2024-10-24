import os
from flask import Flask, render_template, request, send_from_directory
from classifier import classifyImage

MODE = "development"
DEV_SERVER_URL = 'http://localhost:3000/'

app = Flask(__name__)

@app.route('/<path:path>')
def index(path):
    if path == '':
        return send_from_directory('./MAIN', "/index.html")
    else:
        return send_from_directory('./MAIN', path)

@app.route('/classify', methods=['POST'])
def classify():
    
    if (request.files['image']): 
        file = request.files['image']

        result = classifyImage(file)    
        return result

app.run(host='0.0.0.0')