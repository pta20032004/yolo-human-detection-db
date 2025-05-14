#!/usr/bin/env python3
"""
Test script cho hiệu suất embedding với MobileFaceNet
Test performance of face embedding with MobileFaceNet

Usage:
    python test.py --image_path path/to/image.jpg
    python test.py --folder_path path/to/images/
    python test.py --webcam
"""

import os
import time
import argparse
import numpy as np
import cv2
from glob import glob
import statistics
from pathlib import Path

# Import box_detector
from app.box_detector import Detector

def test_single_image(detector, image_path, visualize=True):
    """
    Test embedding extraction trên một ảnh
    
    Args:
        detector: Detector instance
        image_path: Đường dẫn đến ảnh
        visualize: Hiển thị ảnh với bounding boxes
    
    Returns:
        dict: Kết quả test
    """
    print(f"\n=== Testing image: {os.path.basename(image_path)} ===")
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Không thể đọc ảnh {image_path}")
        return None
    
    print(f"Image size: {image.shape}")
    
    # Đo thời gian toàn bộ quá trình
    start_total = time.perf_counter()
    
    # Process frame (detection + embedding)
    person_count, face_count, person_boxes, face_boxes = detector.process_frame(image)
    
    end_total = time.perf_counter()
    total_time = (end_total - start_total) * 1000  # ms
    
    # Thu thập kết quả
    result = {
        "image_path": image_path,
        "person_count": person_count,
        "face_count": face_count,
        "total_time_ms": total_time,
        "avg_time_per_face_ms": total_time / max(face_count, 1),
        "face_boxes": face_boxes
    }
    
    # In kết quả
    print(f"Persons detected: {person_count}")
    print(f"Faces detected: {face_count}")
    print(f"Total processing time: {total_time:.2f} ms")
    print(f"Average time per face: {result['avg_time_per_face_ms']:.2f} ms")
    
    # Kiểm tra embedding
    embedding_count = 0
    for box in face_boxes:
        if len(box) > 3 and box[3] is not None:
            embedding_count += 1
            embedding = np.array(box[3])
            print(f"Embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.4f}")
    
    print(f"Successful embeddings: {embedding_count}/{face_count}")
    
    # Visualization
    if visualize and face_count > 0:
        vis_image = visualize_results(image, person_boxes, face_boxes)
        output_path = f"test_output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, vis_image)
        print(f"Saved visualization to {output_path}")
    
    return result

def visualize_results(image, person_boxes, face_boxes):
    """
    Vẽ bounding boxes lên ảnh
    """
    vis_image = image.copy()
    
    # Vẽ person boxes (màu đỏ)
    for (x1, y1, x2, y2), conf in person_boxes:
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis_image, f"Person {conf:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Vẽ face boxes (màu xanh lá)
    for i, ((x1, y1, x2, y2), conf, emotion, embedding) in enumerate(face_boxes):
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label với confidence và emotion
        label = f"Face {conf:.2f}"
        if emotion and emotion != "Không xác định":
            label += f" - {emotion}"
        
        cv2.putText(vis_image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Hiện thị embedding status
        embed_status = "✓" if embedding is not None else "✗"
        cv2.putText(vis_image, f"Embed: {embed_status}", (x1, y2+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return vis_image

def test_folder(detector, folder_path, max_images=10):
    """
    Test trên multiple images trong folder
    """
    print(f"\n=== Testing folder: {folder_path} ===")
    
    # Tìm ảnh trong folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
        image_files.extend(glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    # Giới hạn số ảnh test
    image_files = image_files[:max_images]
    print(f"Testing {len(image_files)} images...")
    
    results = []
    for img_path in image_files:
        result = test_single_image(detector, img_path, visualize=False)
        if result:
            results.append(result)
    
    # Tính thống kê
    calculate_statistics(results)

def calculate_statistics(results):
    """
    Tính và hiển thị thống kê từ multiple tests
    """
    if not results:
        return
    
    print(f"\n=== STATISTICS ({len(results)} images) ===")
    
    # Gather metrics
    total_times = [r['total_time_ms'] for r in results]
    per_face_times = [r['avg_time_per_face_ms'] for r in results if r['face_count'] > 0]
    face_counts = [r['face_count'] for r in results]
    person_counts = [r['person_count'] for r in results]
    
    # Total processing time stats
    print(f"Total processing time:")
    print(f"  Mean: {statistics.mean(total_times):.2f} ms")
    print(f"  Median: {statistics.median(total_times):.2f} ms")
    print(f"  Min: {min(total_times):.2f} ms")
    print(f"  Max: {max(total_times):.2f} ms")
    
    # Per-face time stats
    if per_face_times:
        print(f"\nPer-face processing time:")
        print(f"  Mean: {statistics.mean(per_face_times):.2f} ms")
        print(f"  Median: {statistics.median(per_face_times):.2f} ms")
        print(f"  Min: {min(per_face_times):.2f} ms")
        print(f"  Max: {max(per_face_times):.2f} ms")
    
    # Detection stats
    print(f"\nDetection statistics:")
    print(f"  Average faces per image: {statistics.mean(face_counts):.1f}")
    print(f"  Average persons per image: {statistics.mean(person_counts):.1f}")
    print(f"  Total faces detected: {sum(face_counts)}")
    
    # Performance assessment
    avg_per_face = statistics.mean(per_face_times) if per_face_times else 0
    print(f"\n=== PERFORMANCE ASSESSMENT ===")
    print(f"Average embedding extraction time: {avg_per_face:.2f} ms/face")
    
    if avg_per_face < 15:
        print("✅ PASSED: Meets <15ms requirement!")
    else:
        print("❌ FAILED: Does not meet <15ms requirement")
        print("Consider optimization or switching to int8 model")

def test_webcam(detector, duration_seconds=30):
    """
    Test real-time performance với webcam
    """
    print(f"\n=== Testing webcam for {duration_seconds} seconds ===")
    print("Press 'q' to quit early")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_times = []
    face_counts = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_start = time.perf_counter()
            person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)
            frame_end = time.perf_counter()
            
            frame_time = (frame_end - frame_start) * 1000
            frame_times.append(frame_time)
            face_counts.append(face_count)
            
            # Visualize
            vis_frame = visualize_results(frame, person_boxes, face_boxes)
            
            # Add performance info
            fps = 1000 / frame_time
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Faces: {face_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Time: {frame_time:.1f}ms", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection & Embedding Test', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Calculate webcam stats
    if frame_times:
        print(f"\n=== WEBCAM PERFORMANCE ===")
        print(f"Frames processed: {len(frame_times)}")
        print(f"Average FPS: {1000/statistics.mean(frame_times):.1f}")
        print(f"Average processing time: {statistics.mean(frame_times):.2f} ms")
        print(f"Average faces per frame: {statistics.mean(face_counts):.1f}")

def test_benchmark_embedding_only(detector, test_image_path, num_runs=100):
    """
    Benchmark chỉ riêng embedding extraction
    """
    print(f"\n=== EMBEDDING-ONLY BENCHMARK ({num_runs} runs) ===")
    
    # Đọc và preprocess ảnh
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Cannot load {test_image_path}")
        return
    
    # Get một face crop để test
    _, _, _, face_boxes = detector.process_frame(image)
    if not face_boxes:
        print("No faces detected in test image")
        return
    
    # Lấy face ROI đầu tiên
    (x1, y1, x2, y2), _, _, _ = face_boxes[0]
    face_roi = image[y1:y2, x1:x2]
    face_roi = detector.extract_face_with_margin(face_roi, (0, 0, x2-x1, y2-y1))
    
    print(f"Testing with face crop of size: {face_roi.shape}")
    
    # Benchmark embedding extraction
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        embedding = detector.get_embedding_from_image(face_roi)
        end = time.perf_counter()
        
        if embedding is not None:
            times.append((end - start) * 1000)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_runs} runs")
    
    # Stats
    if times:
        print(f"\n=== EMBEDDING BENCHMARK RESULTS ===")
        print(f"Successful runs: {len(times)}/{num_runs}")
        print(f"Mean time: {statistics.mean(times):.2f} ms")
        print(f"Median time: {statistics.median(times):.2f} ms")
        print(f"Min time: {min(times):.2f} ms")
        print(f"Max time: {max(times):.2f} ms")
        print(f"Std dev: {statistics.stdev(times):.2f} ms")
        
        # Percentiles
        times_sorted = sorted(times)
        p95 = times_sorted[int(0.95 * len(times_sorted))]
        p99 = times_sorted[int(0.99 * len(times_sorted))]
        print(f"P95: {p95:.2f} ms")
        print(f"P99: {p99:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description='Test MobileFaceNet embedding performance')
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--folder', type=str, help='Path to folder of test images')
    parser.add_argument('--webcam', action='store_true', help='Test with webcam')
    parser.add_argument('--max_images', type=int, default=10, help='Max images to test in folder')
    parser.add_argument('--webcam_duration', type=int, default=30, help='Webcam test duration in seconds')
    parser.add_argument('--benchmark', type=str, help='Benchmark embedding-only with specific image')
    parser.add_argument('--benchmark_runs', type=int, default=100, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing detector...")
    start_init = time.time()
    detector = Detector()
    init_time = time.time() - start_init
    print(f"Detector initialized in {init_time:.2f} seconds")
    
    # Run tests based on arguments
    if args.image:
        test_single_image(detector, args.image)
    
    elif args.folder:
        test_folder(detector, args.folder, args.max_images)
    
    elif args.webcam:
        test_webcam(detector, args.webcam_duration)
    
    elif args.benchmark:
        test_benchmark_embedding_only(detector, args.benchmark, args.benchmark_runs)
    
    else:
        print("No test specified. Use --help for options")
        print("\nExample usage:")
        print("  python test.py --image test_image.jpg")
        print("  python test.py --folder ./test_images/")
        print("  python test.py --webcam")
        print("  python test.py --benchmark test_image.jpg")

if __name__ == "__main__":
    main()