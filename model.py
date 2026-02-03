import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def prepare_image(img_path):
    """Load and preprocess image for prediction"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    """Predict the class of an image"""
    processed_img = prepare_image(img_path)
    preds = model.predict(processed_img)
    decoded = decode_predictions(preds, top=3)[0]
    # Return top 3 predictions
    results = [(cls, desc, float(prob)) for (cls, desc, prob) in decoded]
    return results
  
