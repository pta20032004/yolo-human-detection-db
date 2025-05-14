#!/usr/bin/env python3
"""
Detailed benchmark script cho embedding performance
Focus on pure embedding time và GPU utilization
"""

import os
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
from app.box_detector import Detector

def check_gpu_availability():
    """Check GPU availability và configuration"""
    print("=== GPU INFORMATION ===")
    
    # TensorFlow GPU info
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available to TensorFlow: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
        # Check GPU memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  Memory growth enabled for GPU {i}")
        except:
            print(f"  Cannot set memory growth for GPU {i}")
    
    # Check TFLite delegates
    print(f"\nTensorFlow version: {tf.__version__}")
    
    # List available TFLite delegates
    try:
        delegates = tf.lite.experimental.load_delegate
        print("TFLite delegates available")
    except:
        print("TFLite GPU delegate check failed")

def benchmark_embedding_detailed(detector, test_image_path, num_runs=100):
    """
    Detailed benchmark chỉ embedding với warmup và stats
    """
    print(f"\n=== DETAILED EMBEDDING BENCHMARK ===")
    print(f"Test image: {test_image_path}")
    print(f"Runs: {num_runs}")
    
    # Đọc và prepare face crop
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Cannot load {test_image_path}")
        return
    
    # Get face detection first (không tính thời gian này)
    print("Getting face crop (not counted in timing)...")
    _, _, _, face_boxes = detector.process_frame(image)
    if not face_boxes:
        print("No faces detected in test image")
        return
    
    # Get face ROI
    (x1, y1, x2, y2), _, _, _ = face_boxes[0]
    face_roi = image[y1:y2, x1:x2]
    face_roi = detector.extract_face_with_margin(face_roi, (0, 0, x2-x1, y2-y1))
    
    print(f"Face crop size: {face_roi.shape}")
    
    # Check embedding dimension
    sample_embedding = detector.get_embedding_from_image(face_roi)
    if sample_embedding is not None:
        print(f"Embedding dimension: {sample_embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(sample_embedding):.4f}")
    
    # Warmup runs (không tính)
    print("\nWarming up model...")
    for _ in range(10):
        _ = detector.get_embedding_from_image(face_roi)
    
    # Actual benchmark
    print(f"\nRunning {num_runs} benchmark iterations...")
    
    times = []
    successful_runs = 0
    
    for i in range(num_runs):
        # Precise timing chỉ riêng embedding
        start = time.perf_counter()
        embedding = detector.get_embedding_from_image(face_roi)
        end = time.perf_counter()
        
        if embedding is not None:
            times.append((end - start) * 1000)  # Convert to ms
            successful_runs += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_runs}")
    
    # Calculate statistics
    if times:
        times_array = np.array(times)
        
        print(f"\n=== RESULTS ===")
        print(f"Successful runs: {successful_runs}/{num_runs}")
        print(f"Mean time: {np.mean(times_array):.3f} ms")
        print(f"Median time: {np.median(times_array):.3f} ms")
        print(f"Min time: {np.min(times_array):.3f} ms")
        print(f"Max time: {np.max(times_array):.3f} ms")
        print(f"Std deviation: {np.std(times_array):.3f} ms")
        
        # Percentiles
        p50 = np.percentile(times_array, 50)
        p95 = np.percentile(times_array, 95)
        p99 = np.percentile(times_array, 99)
        
        print(f"P50 (median): {p50:.3f} ms")
        print(f"P95: {p95:.3f} ms") 
        print(f"P99: {p99:.3f} ms")
        
        # Performance assessment
        avg_time = np.mean(times_array)
        print(f"\n=== PERFORMANCE ASSESSMENT ===")
        if avg_time < 5:
            print(f"🚀 EXCELLENT: {avg_time:.3f}ms << 15ms target")
        elif avg_time < 15:
            print(f"✅ GOOD: {avg_time:.3f}ms < 15ms target")
        elif avg_time < 30:
            print(f"⚠️  ACCEPTABLE: {avg_time:.3f}ms > 15ms target")
        else:
            print(f"❌ SLOW: {avg_time:.3f}ms >> 15ms target")
            
        # Estimate FPS với embedding
        estimated_fps = 1000 / avg_time
        print(f"Theoretical embedding FPS: {estimated_fps:.1f}")
        
        return {
            'mean_ms': np.mean(times_array),
            'median_ms': np.median(times_array),
            'p95_ms': p95,
            'p99_ms': p99,
            'success_rate': successful_runs / num_runs
        }
    
    else:
        print("❌ No successful runs!")
        return None

def compare_models():
    """So sánh float32 vs int8 models"""
    print("\n=== MODEL COMPARISON ===")
    
    results = {}
    
    for model_name in ['mobilefacenet_float32.tflite', 'mobilefacenet_int8.tflite']:
        print(f"\nTesting {model_name}...")
        
        # Tạm thời modify detector để load model khác
        # Cách đơn giản: copy và modify _load_face_embedding_model
        
        try:
            # Load model để check info
            model_path = f'./models/{model_name}'
            if os.path.exists(model_path):
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                print(f"  Input shape: {input_details[0]['shape']}")
                print(f"  Input dtype: {input_details[0]['dtype']}")
                print(f"  Output shape: {output_details[0]['shape']}")
                print(f"  Output dtype: {output_details[0]['dtype']}")
                
                # File size
                file_size = os.path.getsize(model_path) / (1024*1024)  # MB
                print(f"  File size: {file_size:.2f} MB")
                
                results[model_name] = {
                    'path': model_path,
                    'input_shape': input_details[0]['shape'],
                    'input_dtype': str(input_details[0]['dtype']),
                    'output_shape': output_details[0]['shape'],
                    'output_dtype': str(output_details[0]['dtype']),
                    'file_size_mb': file_size
                }
            else:
                print(f"  ❌ Model not found: {model_path}")
                
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
    
    return results

def test_model_loading_time():
    """Test thời gian load model"""
    print("\n=== MODEL LOADING TIME TEST ===")
    
    models_to_test = [
        'mobilefacenet_float32.tflite',
        'mobilefacenet_int8.tflite'
    ]
    
    for model_name in models_to_test:
        model_path = f'./models/{model_name}'
        if not os.path.exists(model_path):
            print(f"❌ {model_name} not found")
            continue
        
        # Test load time
        load_times = []
        for i in range(5):
            start = time.perf_counter()
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            end = time.perf_counter()
            load_times.append((end - start) * 1000)
        
        avg_load_time = np.mean(load_times)
        print(f"{model_name}: {avg_load_time:.2f} ms avg load time")

def create_test_detector_with_model(model_name):
    """Tạo detector với model khác"""
    # Đây là hack để test model khác
    # Trong thực tế nên có config để switch model
    
    # Modify path trong file tạm thời
    original_file = 'app/box_detector.py'
    backup_file = 'app/box_detector.py.backup'
    
    # Read original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Backup
    with open(backup_file, 'w') as f:
        f.write(content)
    
    # Replace model name
    modified_content = content.replace(
        'mobilefacenet_float32.tflite',
        model_name
    )
    
    # Write modified file
    with open(original_file, 'w') as f:
        f.write(modified_content)
    
    try:
        # Import the modified detector
        import importlib
        import app.box_detector
        importlib.reload(app.box_detector)
        detector = app.box_detector.Detector()
        return detector
    finally:
        # Restore original file
        with open(backup_file, 'r') as f:
            original_content = f.read()
        with open(original_file, 'w') as f:
            f.write(original_content)
        os.remove(backup_file)

def main():
    parser = argparse.ArgumentParser(description='Detailed embedding performance test')
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--check_gpu', action='store_true', help='Check GPU availability')
    parser.add_argument('--compare_models', action='store_true', help='Compare float32 vs int8')
    parser.add_argument('--test_loading', action='store_true', help='Test model loading time')
    
    args = parser.parse_args()
    
    # GPU check
    if args.check_gpu:
        check_gpu_availability()
    
    # Model comparison
    if args.compare_models:
        model_info = compare_models()
    
    # Loading time test
    if args.test_loading:
        test_model_loading_time()
    
    # Main benchmark with current model (float32)
    print("\n=== CURRENT MODEL BENCHMARK (float32) ===")
    detector = Detector()
    result = benchmark_embedding_detailed(detector, args.image, args.runs)
    
    # Optional: Test int8 model
    print("\n=== INT8 MODEL BENCHMARK ===")
    try:
        if os.path.exists('./models/mobilefacenet_int8.tflite'):
            detector_int8 = create_test_detector_with_model('mobilefacenet_int8.tflite')
            result_int8 = benchmark_embedding_detailed(detector_int8, args.image, args.runs)
            
            # Compare results
            if result and result_int8:
                print(f"\n=== COMPARISON ===")
                print(f"Float32 mean: {result['mean_ms']:.3f} ms")
                print(f"Int8 mean: {result_int8['mean_ms']:.3f} ms")
                print(f"Speedup: {result['mean_ms'] / result_int8['mean_ms']:.2f}x")
        else:
            print("Int8 model not found, skipping comparison")
    except Exception as e:
        print(f"Error testing int8 model: {e}")

if __name__ == "__main__":
    main()