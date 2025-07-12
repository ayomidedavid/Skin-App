from keras.models import load_model
from keras.preprocessing import image
try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow is required for image loading. Please install it with 'pip install Pillow'.")
import numpy as np
import os

label_names = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic nevi',
    5: 'Vascular lesions',
    6: 'Melanoma'
}

model_a_classes = [4, 6, 2]  # nv, mel, bkl
model_b_classes = [1, 0, 5, 3]  # bcc, akiec, vasc, df

# Get the project root (one level above 'src')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SkinClassifier:
    def __init__(self, model1_path, model2_path):
        # model1_path and model2_path are already absolute paths
        print("Looking for model1 at:", model1_path)
        print("Looking for model2 at:", model2_path)
        if not os.path.exists(model1_path):
            raise FileNotFoundError(f"Model file not found: {model1_path}")
        if not os.path.exists(model2_path):
            raise FileNotFoundError(f"Model file not found: {model2_path}")
        self.model1 = load_model(model1_path)
        self.model2 = load_model(model2_path)
        self.model_a_classes = model_a_classes
        self.model_b_classes = model_b_classes

    def preprocess_image(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict(self, img_path):
        processed_image = self.preprocess_image(img_path)
        # Model A prediction
        predictions1 = self.model1.predict(processed_image)
        idx1 = np.argmax(predictions1, axis=1)[0]
        orig_class1 = self.model_a_classes[idx1]
        prob1 = predictions1[0][idx1]
        # Model B prediction
        predictions2 = self.model2.predict(processed_image)
        idx2 = np.argmax(predictions2, axis=1)[0]
        orig_class2 = self.model_b_classes[idx2]
        prob2 = predictions2[0][idx2]
        # Compare and return the highest probability result
        if prob1 >= prob2:
            label = label_names.get(orig_class1, str(orig_class1))
            return label, prob1, 'model_a'
        else:
            label = label_names.get(orig_class2, str(orig_class2))
            return label, prob2, 'model_b'

def get_skin_classifier():
    model1_path = "C:/Users/FALOWO PC/Downloads/skin-app/skin-classification-system/Model_A_Multiclass.keras"
    model2_path = "C:/Users/FALOWO PC/Downloads/skin-app/skin-classification-system/Model_B_Multiclass.keras"
    
    print("DEBUG: model1_path =", model1_path)
    print("DEBUG: model2_path =", model2_path)

    return SkinClassifier(model1_path, model2_path)


def predict_skin_condition(img_path):
    classifier = get_skin_classifier()
    return classifier.predict(img_path)

#Example Flask usage:
from flask import Flask, request, render_template
app = Flask(__name__)
classifier = get_skin_classifier()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    file_path = os.path.join('static', 'uploads', file.filename)
    file.save(file_path)
    result = classifier.predict(file_path)
    if isinstance(result, tuple) and len(result) == 3:
        label, prob, model_used = result
        return render_template('results.html', label=label, probability=prob, model=model_used, image=file.filename)
    else:
        return 'Prediction error: unexpected result format', 500