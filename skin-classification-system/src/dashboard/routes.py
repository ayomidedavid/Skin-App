from flask import Blueprint, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from model.predict import predict_skin_condition

dashboard_bp = Blueprint('dashboard', __name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@dashboard_bp.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('auth.login'))
    username = session['username']
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            # Ensure the upload folder exists and is a directory
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            elif not os.path.isdir(UPLOAD_FOLDER):
                raise RuntimeError(f"{UPLOAD_FOLDER} exists and is not a directory. Please remove or rename it.")
            file.save(file_path)
            predicted_class, probability, model_used = predict_skin_condition(file_path)
            return redirect(url_for('dashboard.result', image_filename=filename, predicted_class=predicted_class, probability=probability, model=model_used))
    return render_template('dashboard.html', username=username)

@dashboard_bp.route('/result')
def result():
    image_filename = request.args.get('image_filename')
    predicted_class = request.args.get('predicted_class')
    probability = request.args.get('probability')
    return render_template('results.html', image_filename=image_filename, predicted_class=predicted_class, probability=probability)

@dashboard_bp.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('dashboard.dashboard'))

@dashboard_bp.route('/upload-model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'POST':
        if 'model_file' not in request.files:
            return "No file part", 400
        file = request.files['model_file']
        if file.filename == '':
            return "No selected file", 400
        if file:
            # Save as Model_A_Multiclass.h5 in the correct directory
            model_path = os.path.join('skin-classification-system', 'Model_A_Multiclass.h5')
            file.save(model_path)
            return "Model uploaded successfully!"
    return '''
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="model_file" required>
            <button type="submit">Upload Model</button>
        </form>
    '''