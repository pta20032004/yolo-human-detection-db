#!/usr/bin/env python3
"""
Simple Backend Test
Test cơ bản cho API endpoints
"""

import requests
import cv2
import numpy as np
import time

# Config
SERVER_URL = "http://localhost:8000"

def create_test_image():
    """Tạo ảnh test đơn giản"""
    # Tạo ảnh 640x480 màu xanh
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # Vẽ hình chữ nhật giả lập khuôn mặt
    cv2.rectangle(img, (250, 150), (390, 330), (200, 180, 160), -1)
    # Vẽ mắt
    cv2.circle(img, (290, 220), 15, (50, 50, 50), -1)
    cv2.circle(img, (350, 220), 15, (50, 50, 50), -1)
    
    return img

def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"✅ Health: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return False

def test_process_frame():
    """Test process_frame endpoint"""
    print("Testing /process_frame...")
    try:
        # Tạo ảnh test
        img = create_test_image()
        _, img_encoded = cv2.imencode('.jpg', img)
        
        # Gửi request
        files = {'file': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
        start_time = time.time()
        response = requests.post(f"{SERVER_URL}/process_frame", files=files)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            response_time = (end_time - start_time) * 1000
            print(f"✅ Process frame: {response_time:.1f}ms")
            print(f"   Persons: {data.get('persons', 0)}")
            print(f"   Faces: {data.get('faces', 0)}")
            return True
        else:
            print(f"❌ Process frame failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Process frame error: {e}")
        return False

def test_static_files():
    """Test static file serving"""
    print("Testing static files...")
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            print(f"✅ Static files: OK")
            return True
        else:
            print(f"❌ Static files failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Static files error: {e}")
        return False

def test_realtime():
    """Test real-time processing với webcam"""
    print("Testing real-time processing (10 seconds)...")
    print("Press 'q' to quit early")
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        start_time = time.time()
        frame_count = 0
        total_processing_time = 0
        
        while time.time() - start_time < 10:  # Test 10 giây
            ret, frame = cap.read()
            if not ret:
                break
            
            # Gửi frame đến server
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            frame_start = time.time()
            response = requests.post(f"{SERVER_URL}/process_frame", files=files, timeout=1)
            frame_end = time.time()
            
            if response.status_code == 200:
                frame_time = (frame_end - frame_start) * 1000
                total_processing_time += frame_time
                frame_count += 1
                
                # Hiển thị kết quả lên frame
                data = response.json()
                cv2.putText(frame, f"FPS: {1000/frame_time:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Faces: {data.get('faces', 0)}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Real-time Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_fps = frame_count / 10
            avg_latency = total_processing_time / frame_count
            print(f"✅ Real-time test completed")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Average latency: {avg_latency:.1f}ms")
            return True
        else:
            print("❌ No frames processed")
            return False
            
    except Exception as e:
        print(f"❌ Real-time test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Backend Test Suite ===")
    
    tests = [
        ("Health Check", test_health),
        ("Static Files", test_static_files), 
        ("Process Frame", test_process_frame),
        ("Real-time Processing", test_realtime)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        if test_func():
            passed += 1
    
    print(f"\n=== Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.0f}%")

if __name__ == "__main__":
    main()