"""
Package quản lý và nhận diện khuôn mặt sử dụng ChromaDB, hỗ trợ GPU
"""

# Export các class chính để sử dụng dễ dàng
from .chroma_manager import ChromaFaceDB
from .face_recognizer import FaceRecognizer

__all__ = [
    'ChromaFaceDB', 
    'FaceRecognizer',
]