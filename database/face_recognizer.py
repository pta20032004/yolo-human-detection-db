import numpy as np
import cv2
import time
import logging
import uuid
import datetime
from .chroma_manager import ChromaFaceDB
from app.box_detector import Detector

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger('FaceRecognizer')

class FaceRecognizer:
    """
    Nhận diện khuôn mặt sử dụng embedding và ChromaDB
    """
    def __init__(self, db_path="./face_db/chroma_data", threshold=0.7, detector_type="insightface"):
        """
        Khởi tạo FaceRecognizer
        
        Args:
            db_path (str): Đường dẫn đến thư mục ChromaDB
            threshold (float): Ngưỡng tương đồng (0-1)
            detector_type (str): Loại detector ('insightface' hoặc 'yolo')
        """
        logger.info(f"Khởi tạo FaceRecognizer với db_path={db_path}, threshold={threshold}")
        
        # Khởi tạo Detector từ box_detector
        self.detector = Detector()
        
        # Khởi tạo ChromaDB
        start_time = time.time()
        self.face_db = ChromaFaceDB(db_path)
        logger.info(f"Khởi tạo ChromaDB trong {time.time() - start_time:.2f}s")
        
        self.threshold = threshold
        self.performance_stats = {
            "embedding_time": [],
            "search_time": [],
            "total_time": []
        }
        
    def recognize_from_file(self, image_path, top_k=1):
        """
        Nhận diện khuôn mặt từ file ảnh
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            top_k (int): Số lượng kết quả trả về
            
        Returns:
            dict: Kết quả nhận diện hoặc None nếu không nhận diện được
        """
        # Đo thời gian tổng
        start_total = time.time()
        
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
            return []
        
        # Tiếp tục xử lý với ảnh đã đọc
        return self.recognize_from_image(img, top_k)
        
    def recognize_from_image(self, img, top_k=1, max_faces=5):
        """
        Nhận diện nhiều khuôn mặt từ ảnh
        
        Args:
            img (np.ndarray): Ảnh dạng numpy array (BGR)
            top_k (int): Số lượng kết quả trả về cho mỗi khuôn mặt
            max_faces (int): Số lượng khuôn mặt tối đa xử lý
            
        Returns:
            list: Danh sách kết quả nhận diện
        """
        # Đo thời gian tổng
        start_total = time.time()
        
        # Đo thời gian tạo embeddings
        start_embed = time.time()
        _, face_count, _, face_boxes = self.detector.process_frame(img)
        embed_time = time.time() - start_embed
        
        if face_count == 0:
            # Cập nhật thống kê
            total_time = time.time() - start_total
            self.performance_stats["embedding_time"].append(embed_time)
            self.performance_stats["search_time"].append(0)
            self.performance_stats["total_time"].append(total_time)
            return []
        
        # Đo thời gian tìm kiếm
        start_search = time.time()
        
        # Nhận diện từng khuôn mặt
        results = []
        for face_data in face_boxes[:max_faces]:
            # face_data là tuple (coords, conf, emotion, embedding)
            coords = face_data[0]
            conf = face_data[1]
            emotion = face_data[2]
            embedding = face_data[3]
            
            if embedding is None:
                continue
            
            # Tìm kiếm trong database
            matches = self.face_db.search_faces(embedding, top_k, self.threshold)
            
            result = {
                "bbox": coords,
                "score": conf,
                "matches": matches
            }
            
            # Thêm cảm xúc nếu có
            if emotion:
                result["emotion"] = emotion
            
            results.append(result)
        
        search_time = time.time() - start_search
        
        # Tính thời gian tổng
        total_time = time.time() - start_total
        
        # Cập nhật thống kê
        self.performance_stats["embedding_time"].append(embed_time)
        self.performance_stats["search_time"].append(search_time)
        self.performance_stats["total_time"].append(total_time)
        
        # Log hiệu năng
        logger.debug(f"Nhận diện từ ảnh: embed={embed_time:.3f}s, search={search_time:.3f}s, total={total_time:.3f}s, faces={len(results)}")
        
        return results
    
    def register_face(self, image_path, name, user_id=None, metadata=None):
        """
        Đăng ký khuôn mặt mới vào database
        
        Args:
            image_path (str): Đường dẫn đến file ảnh
            name (str): Tên người
            user_id (str, optional): ID người dùng
            metadata (dict, optional): Metadata bổ sung
            
        Returns:
            str: ID của khuôn mặt đã đăng ký hoặc None nếu thất bại
        """
        
        try:
            # Đọc ảnh
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
                return None

            # Tạo embedding
            start_embed = time.time()
            _, face_count, _, face_boxes = self.detector.process_frame(img)
            embed_time = time.time() - start_embed
            if face_count == 0:
                logger.error(f"Không tìm thấy khuôn mặt trong ảnh: {image_path}")
                return None
            
            # Lấy embedding từ khuôn mặt đầu tiên
            embedding = face_boxes[0][3]
            if embedding is None:
                logger.error(f"Không thể tạo embedding từ ảnh: {image_path}")
                return None
                
            # Chuẩn bị metadata
            if user_id is None:
                user_id = f"user_{str(uuid.uuid4())[:8]}"
                
            current_time = datetime.datetime.now().isoformat()
            
            face_metadata = {
                "name": name,
                "user_id": user_id,
                "created_at": current_time,
                "updated_at": current_time,
                "embed_time": embed_time
            }
            
            # Thêm metadata bổ sung nếu có
            if metadata:
                face_metadata["metadata"] = metadata
            
            # Thêm vào database
            start_time = time.time()
            face_id = self.face_db.add_face(embedding, face_metadata, user_id)
            db_time = time.time() - start_time
            
            logger.info(f"Đã đăng ký {name} (ID: {user_id}) - Thời gian: embed={embed_time:.3f}s, db={db_time:.3f}s")
            
            return face_id
        
        except Exception as e:
            logger.error(f"Lỗi khi đăng ký khuôn mặt: {str(e)}")
            return None
        
    def get_performance_stats(self):
        """
        Lấy thống kê hiệu năng
        
        Returns:
            dict: Thống kê hiệu năng
        """
        if not self.performance_stats["total_time"]:
            return {"message": "Chưa có dữ liệu hiệu năng"}
        
        # Tính trung bình
        avg_embed = sum(self.performance_stats["embedding_time"]) / len(self.performance_stats["embedding_time"])
        avg_search = sum(self.performance_stats["search_time"]) / len(self.performance_stats["search_time"])
        avg_total = sum(self.performance_stats["total_time"]) / len(self.performance_stats["total_time"])
        
        # Lấy giá trị tốt nhất (nhanh nhất)
        best_embed = min(self.performance_stats["embedding_time"])
        best_search = min(self.performance_stats["search_time"])
        best_total = min(self.performance_stats["total_time"])
        
        
        return {
            "samples": len(self.performance_stats["total_time"]),
            "avg_embedding_time": avg_embed,
            "avg_search_time": avg_search,
            "avg_total_time": avg_total,
            "best_embedding_time": best_embed,
            "best_search_time": best_search,
            "best_total_time": best_total,
            "avg_fps": 1.0 / avg_total
        }