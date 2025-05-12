from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.box_detector import Detector
import numpy as np
import cv2
import base64
from pathlib import Path
from database.face_recognizer import FaceRecognizer

app = FastAPI()
detector = Detector()
face_recognizer = FaceRecognizer()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoint for processing frames
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    # Đọc nội dung file ảnh / Read image file content
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Xử lý khung hình / Process the frame
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
    
    # Trả về kết quả / Return results
    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf) in person_boxes
        ],
        "face_boxes": [
            {"coords": coords, "confidence": conf, "emotion": emotion}
            for (coords, conf, emotion, _) in face_boxes
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Mount the static files directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Serve index.html at the root
@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

#Test:
# API endpoint for registering faces from a folder
@app.post("/register_folder")
async def register_folder(folder_path: str = Form(...)):
    """
    Đăng ký tất cả ảnh trong thư mục vào database
    """
    if not os.path.exists(folder_path):
        return {"error": f"Thư mục không tồn tại: {folder_path}"}
    
    # Lấy danh sách ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
        image_files.extend(list(Path(folder_path).glob(f"*{ext.upper()}")))
    
    if not image_files:
        return {"error": "Không tìm thấy ảnh trong thư mục"}
    
    success_count = 0
    fail_count = 0
    for img_path in image_files:
        name = img_path.stem
        try:
            face_id = face_recognizer.register_face(str(img_path), name)
            if face_id:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            continue
    
    return {
        "message": f"Đã đăng ký {success_count}/{len(image_files)} khuôn mặt",
        "success_count": success_count,
        "fail_count": fail_count
    }

