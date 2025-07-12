from flask import Blueprint, render_template, request
from model.predict import predict_skin_condition
import os

results_bp = Blueprint('results', __name__)

@results_bp.route('/results', methods=['POST'])
def results():
    if 'image' not in request.files:
        return "No file part", 400
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        # Save the uploaded image
        upload_folder = os.path.join('static', 'uploads')
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        # Predict the skin condition
        predicted_class, probability = predict_skin_condition(file_path)
        
        return render_template('results.html', image=file.filename, predicted_class=predicted_class, probability=probability)

@results_bp.route('/results', methods=['GET'])
def results_get():
    # This route handles GET requests for /results, e.g., after redirect from dashboard
    filename = request.args.get('filename')
    predicted_class = request.args.get('predicted_class')
    probability = request.args.get('probability')
    return render_template('results.html', image=filename, predicted_class=predicted_class, probability=probability)