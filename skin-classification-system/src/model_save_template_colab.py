"""
Template: How to train and save a Keras model in Google Colab for compatibility with Keras 3.x and TensorFlow 2.x+.
This script is for use in Google Colab. After running, download the .keras file to your local machine for use in your Flask app.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# ========== Step 1: Load and preprocess the dataset ==========
CSV_PATH = '/content/drive/MyDrive/SKindataset/hmnist_28_28_RGB.csv'  # Update path if needed
df = pd.read_csv(CSV_PATH)
X_full = df.drop(columns=['label']).values
y_full = df['label'].values

# ========== Step 2: Select classes for your model ==========
# Example for Model A (3 classes)
model_classes = [4, 6, 2]  # Change as needed
label_map = {4: 0, 6: 1, 2: 2}  # Map original labels to 0,1,2

mask = np.isin(y_full, model_classes)
X = X_full[mask]
y = y_full[mask]

# ========== Step 3: Oversample ==========
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
y_mapped = np.array([label_map[label] for label in y])

# ========== Step 4: Preprocess for ResNet50 ==========
IMG_SIZE = 224
X = X.reshape(-1, 28, 28, 3).astype(np.float32)
X = (X - np.mean(X)) / np.std(X)
X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()

# ========== Step 5: Train/val/test split ==========
X_train, X_temp, y_train, y_temp = train_test_split(X, y_mapped, test_size=0.3, stratify=y_mapped, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# ========== Step 6: Build the model ==========
def build_multiclass_model(num_classes):
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_multiclass_model(num_classes=len(label_map))

# ========== Step 7: Train ==========
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# ========== Step 8: Save in .keras format ==========
model.save('/content/Model_A_Multiclass.keras')  # Change filename as needed
print('Model saved as /content/Model_A_Multiclass.keras')

# ========== Step 9: Download the model ==========
from google.colab import files
files.download('/content/Model_A_Multiclass.keras')


# ========== Model B Training and Saving ==========
print("\n\n========== Training and Saving Model B ==========")
model_classes = [1, 0, 5, 3]  # Classes for Model B
label_map = {1: 0, 0: 1, 5: 2, 3: 3}  # Map original labels to 0,1,2,3

mask = np.isin(y_full, model_classes)
X = X_full[mask]
y = y_full[mask]

ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
y_mapped = np.array([label_map[label] for label in y])

X = X.reshape(-1, 28, 28, 3).astype(np.float32)
X = (X - np.mean(X)) / np.std(X)
X = tf.image.resize(X, (IMG_SIZE, IMG_SIZE)).numpy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y_mapped, test_size=0.3, stratify=y_mapped, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

model = build_multiclass_model(num_classes=4)

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

model.save('/content/Model_B_Multiclass.keras')
print('Model B saved as /content/Model_B_Multiclass.keras')

from google.colab import files
files.download('/content/Model_B_Multiclass.keras')
