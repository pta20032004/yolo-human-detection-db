from ultralytics import YOLO
import tensorflow as tf
from .config import PERSON_MODEL_PATH, FACE_MODEL_PATH, EMOTION_MODEL_PATH

def load_models():
    """Tải mô hình YOLO cho nhận diện người và khuôn mặt và mô hình TFLite cho nhận diện cảm xúc
    / Load YOLO models for person and face detection and TFLite model for emotion recognition"""
    
    # Tải mô hình YOLO / Load YOLO models
    person_model = YOLO(PERSON_MODEL_PATH)   # Mô hình nhận diện người / Person model
    face_model = YOLO(FACE_MODEL_PATH)       # Mô hình nhận diện khuôn mặt / Face model
    
    # Tải mô hình nhận diện cảm xúc TensorFlow Lite / Load TFLite emotion recognition model
    emotion_interpreter = tf.lite.Interpreter(model_path=EMOTION_MODEL_PATH)
    emotion_interpreter.allocate_tensors()
    
    return person_model, face_model, emotion_interpreter

def get_emotion_model_details(interpreter):
    """Lấy thông tin đầu vào và đầu ra của mô hình cảm xúc
    / Get input and output details for the emotion model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details