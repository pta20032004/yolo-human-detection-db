"""
Simple Test Script
Test đăng ký khuôn mặt và nhận diện real-time
Updated with enhanced features
"""

import os
import sys
import cv2
import time
import numpy as np

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.face_registration_tool import FaceRegistrationTool
from database.chroma_manager import ChromaFaceDB
from app.box_detector import Detector

class SimpleTest:
    """Simple test suite for face registration and recognition"""
    
    def __init__(self):
        """Initialize test components"""
        print("🚀 Initializing Simple Test Suite...")
        
        # Initialize components
        self.registration_tool = FaceRegistrationTool()
        self.face_db = ChromaFaceDB()
        self.detector = Detector()
        
        print("✅ All components ready!")
    
    def test_registration(self, image_path, person_name):
        """
        Test face registration
        
        Args:
            image_path (str): Path to image
            person_name (str): Name of person
        
        Returns:
            bool: Success status
        """
        print(f"\n📝 TESTING FACE REGISTRATION")
        print(f"Image: {image_path}")
        print(f"Name: {person_name}")
        
        # Register face
        face_id = self.registration_tool.register_single_face(image_path, person_name)
        
        if face_id:
            print(f"✅ Registration successful! Face ID: {face_id}")
            return True
        else:
            print("❌ Registration failed!")
            return False
    
    def test_people_folder_registration(self, people_folder_path, max_per_person=None):
        """
        Test multiple people registration from folder structure
        
        Args:
            people_folder_path (str): Path to folder containing person folders
            max_per_person (int): Maximum images per person
        
        Returns:
            dict: Registration results
        """
        print(f"\n👥 TESTING PEOPLE FOLDER REGISTRATION")
        print(f"Folder: {people_folder_path}")
        print(f"Max per person: {max_per_person if max_per_person else 'No limit'}")
        
        # Reset database option
        reset = input("Reset database before registration? (y/N): ").strip().lower()
        if reset == 'y':
            print("🗑️  Resetting database...")
            self.registration_tool.reset_database()
        
        # Register people
        results = self.registration_tool.register_people_folders(people_folder_path, max_per_person)
        
        return results
    
    def test_recognition_image(self, test_image_path):
        """
        Test face recognition on single image
        
        Args:
            test_image_path (str): Path to test image
        
        Returns:
            list: Recognition results
        """
        print(f"\n🔍 TESTING FACE RECOGNITION")
        print(f"Test image: {test_image_path}")
        
        # Read image
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Cannot read test image!")
            return None
        
        # Detect faces and get embeddings
        start_time = time.time()
        person_count, face_count, person_boxes, face_boxes = self.detector.process_frame(image)
        detection_time = time.time() - start_time
        
        print(f"Detection time: {detection_time:.3f}s")
        print(f"Faces detected: {face_count}")
        
        if face_count == 0:
            print("❌ No faces detected!")
            return None
        
        # Search for each detected face
        recognition_results = []
        for i, (coords, conf, emotion, embedding) in enumerate(face_boxes):
            print(f"\nFace {i+1}:")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Emotion: {emotion}")
            
            if embedding is not None:
                # Convert to numpy array if needed
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Search in database
                start_search = time.time()
                matches = self.face_db.search_faces(embedding, top_k=3, threshold=0.7)
                search_time = time.time() - start_search
                
                print(f"  Search time: {search_time*1000:.1f}ms")
                print(f"  Matches found: {len(matches)}")
                
                if matches:
                    for j, match in enumerate(matches):
                        metadata = match['metadata']
                        similarity = match['similarity']
                        name = metadata.get('name', 'Unknown')
                        print(f"    {j+1}. {name} (similarity: {similarity:.3f})")
                else:
                    print("    No matches found")
                
                recognition_results.append({
                    'face_id': i,
                    'coordinates': coords,
                    'confidence': conf,
                    'emotion': emotion,
                    'matches': matches,
                    'search_time_ms': search_time * 1000
                })
            else:
                print("  ❌ Failed to extract embedding")
        
        return recognition_results
    
    def test_recognition_realtime(self, duration_seconds=30):
        """
        Test real-time face recognition with webcam
        
        Args:
            duration_seconds (int): Test duration in seconds
        """
        print(f"\n📹 TESTING REAL-TIME RECOGNITION")
        print(f"Duration: {duration_seconds} seconds")
        print("Press 'q' to quit early, 'r' to register current face, 's' to save screenshot")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        total_detection_time = 0
        total_search_time = 0
        recognition_stats = {"recognized": 0, "unknown": 0}
        
        print("✅ Webcam started. Real-time recognition running...")
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection and recognition
            detect_start = time.time()
            person_count, face_count, person_boxes, face_boxes = self.detector.process_frame(frame)
            detect_time = time.time() - detect_start
            total_detection_time += detect_time
            
            # Draw person boxes (red)
            for (x1, y1, x2, y2), conf in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Process each face
            for (x1, y1, x2, y2), conf, emotion, embedding in face_boxes:
                # Draw face box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Search for face if embedding exists
                face_label = f"Face {conf:.2f}"
                if emotion:
                    face_label += f" - {emotion}"
                
                if embedding is not None:
                    # Convert to numpy array if needed
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    # Search in database
                    search_start = time.time()
                    matches = self.face_db.search_faces(embedding, top_k=1, threshold=0.75)
                    search_time = time.time() - search_start
                    total_search_time += search_time
                    
                    # Add recognition result to label
                    if matches:
                        best_match = matches[0]
                        name = best_match['metadata'].get('name', 'Unknown')
                        similarity = best_match['similarity']
                        face_label = f"{name} ({similarity:.2f})"
                        # Use blue color for recognized faces
                        color = (255, 0, 0)
                        recognition_stats["recognized"] += 1
                    else:
                        face_label += " - Unknown"
                        color = (0, 255, 0)
                        recognition_stats["unknown"] += 1
                else:
                    color = (0, 255, 0)
                
                # Draw label
                cv2.putText(frame, face_label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add performance info
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection: {detect_time*1000:.1f}ms", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add recognition stats
            cv2.putText(frame, f"Recognized: {recognition_stats['recognized']}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Unknown: {recognition_stats['unknown']}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Face Recognition Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Register current frame
                self._register_current_frame(frame)
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print(f"\n📊 REAL-TIME TEST STATISTICS")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Average detection time: {total_detection_time/frame_count*1000:.1f}ms")
        print(f"Average search time: {total_search_time/max(1, frame_count)*1000:.1f}ms")
        print(f"Recognition stats: {recognition_stats['recognized']} recognized, {recognition_stats['unknown']} unknown")
        
        # Recognition rate
        total_recognitions = recognition_stats['recognized'] + recognition_stats['unknown']
        if total_recognitions > 0:
            recognition_rate = recognition_stats['recognized'] / total_recognitions * 100
            print(f"Recognition rate: {recognition_rate:.1f}%")
    
    def _register_current_frame(self, frame):
        """Register face from current frame"""
        person_name = input("\nEnter name for registration: ")
        if person_name:
            # Save temporary frame
            temp_path = "temp_registration.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Register face
            face_id = self.registration_tool.register_single_face(temp_path, person_name)
            
            # Cleanup
            os.remove(temp_path)
            
            if face_id:
                print(f"✅ Registered {person_name} successfully!")
            else:
                print(f"❌ Failed to register {person_name}")
    
    def test_database_status(self):
        """Test database status and statistics"""
        print(f"\n📊 DATABASE STATUS"""
Simple Test Script
Test đăng ký khuôn mặt và nhận diện real-time
"""

import os
import sys
import cv2
import time
import numpy as np

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.face_registration_tool import FaceRegistrationTool
from database.chroma_manager import ChromaFaceDB
from app.box_detector import Detector

class SimpleTest:
    """Simple test suite for face registration and recognition"""
    
    def __init__(self):
        """Initialize test components"""
        print("🚀 Initializing Simple Test Suite...")
        
        # Initialize components
        self.registration_tool = FaceRegistrationTool()
        self.face_db = ChromaFaceDB()
        self.detector = Detector()
        
        print("✅ All components ready!")
    
    def test_registration(self, image_path, person_name):
        """
        Test face registration
        
        Args:
            image_path (str): Path to image
            person_name (str): Name of person
        
        Returns:
            bool: Success status
        """
        print(f"\n📝 TESTING FACE REGISTRATION")
        print(f"Image: {image_path}")
        print(f"Name: {person_name}")
        
        # Register face
        face_id = self.registration_tool.register_single_face(image_path, person_name)
        
        if face_id:
            print(f"✅ Registration successful! Face ID: {face_id}")
            return True
        else:
            print("❌ Registration failed!")
            return False
    
    def test_recognition_image(self, test_image_path):
        """
        Test face recognition on single image
        
        Args:
            test_image_path (str): Path to test image
        
        Returns:
            list: Recognition results
        """
        print(f"\n🔍 TESTING FACE RECOGNITION")
        print(f"Test image: {test_image_path}")
        
        # Read image
        image = cv2.imread(test_image_path)
        if image is None:
            print("❌ Cannot read test image!")
            return None
        
        # Detect faces and get embeddings
        start_time = time.time()
        person_count, face_count, person_boxes, face_boxes = self.detector.process_frame(image)
        detection_time = time.time() - start_time
        
        print(f"Detection time: {detection_time:.3f}s")
        print(f"Faces detected: {face_count}")
        
        if face_count == 0:
            print("❌ No faces detected!")
            return None
        
        # Search for each detected face
        recognition_results = []
        for i, (coords, conf, emotion, embedding) in enumerate(face_boxes):
            print(f"\nFace {i+1}:")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Emotion: {emotion}")
            
            if embedding is not None:
                # Convert to numpy array if needed
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                # Search in database
                start_search = time.time()
                matches = self.face_db.search_faces(embedding, top_k=3, threshold=0.7)
                search_time = time.time() - start_search
                
                print(f"  Search time: {search_time*1000:.1f}ms")
                print(f"  Matches found: {len(matches)}")
                
                if matches:
                    for j, match in enumerate(matches):
                        metadata = match['metadata']
                        similarity = match['similarity']
                        name = metadata.get('name', 'Unknown')
                        print(f"    {j+1}. {name} (similarity: {similarity:.3f})")
                else:
                    print("    No matches found")
                
                recognition_results.append({
                    'face_id': i,
                    'coordinates': coords,
                    'confidence': conf,
                    'emotion': emotion,
                    'matches': matches,
                    'search_time_ms': search_time * 1000
                })
            else:
                print("  ❌ Failed to extract embedding")
        
        return recognition_results
    
    def test_recognition_realtime(self, duration_seconds=30):
        """
        Test real-time face recognition with webcam
        
        Args:
            duration_seconds (int): Test duration in seconds
        """
        print(f"\n📹 TESTING REAL-TIME RECOGNITION")
        print(f"Duration: {duration_seconds} seconds")
        print("Press 'q' to quit early, 'r' to register current face")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        total_detection_time = 0
        total_search_time = 0
        
        print("✅ Webcam started. Real-time recognition running...")
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detection and recognition
            detect_start = time.time()
            person_count, face_count, person_boxes, face_boxes = self.detector.process_frame(frame)
            detect_time = time.time() - detect_start
            total_detection_time += detect_time
            
            # Draw person boxes (red)
            for (x1, y1, x2, y2), conf in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Process each face
            for (x1, y1, x2, y2), conf, emotion, embedding in face_boxes:
                # Draw face box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Search for face if embedding exists
                face_label = f"Face {conf:.2f}"
                if emotion:
                    face_label += f" - {emotion}"
                
                if embedding is not None:
                    # Convert to numpy array if needed
                    if isinstance(embedding, list):
                        embedding = np.array(embedding)
                    
                    # Search in database
                    search_start = time.time()
                    matches = self.face_db.search_faces(embedding, top_k=1, threshold=0.75)
                    search_time = time.time() - search_start
                    total_search_time += search_time
                    
                    # Add recognition result to label
                    if matches:
                        best_match = matches[0]
                        name = best_match['metadata'].get('name', 'Unknown')
                        similarity = best_match['similarity']
                        face_label = f"{name} ({similarity:.2f})"
                        # Use blue color for recognized faces
                        color = (255, 0, 0)
                    else:
                        face_label += " - Unknown"
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 0)
                
                # Draw label
                cv2.putText(frame, face_label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add performance info
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {face_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection: {detect_time*1000:.1f}ms", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Real-time Face Recognition Test', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Register current frame
                self._register_current_frame(frame)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print(f"\n📊 REAL-TIME TEST STATISTICS")
        print(f"Duration: {time.time() - start_time:.1f}s")
        print(f"Frames processed: {frame_count}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Average detection time: {total_detection_time/frame_count*1000:.1f}ms")
        print(f"Average search time: {total_search_time/max(1, frame_count):.1f}ms")
    
    def _register_current_frame(self, frame):
        """Register face from current frame"""
        person_name = input("Enter name for registration: ")
        if person_name:
            # Save temporary frame
            temp_path = "temp_registration.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Register face
            face_id = self.registration_tool.register_single_face(temp_path, person_name)
            
            # Cleanup
            os.remove(temp_path)
            
            if face_id:
                print(f"✅ Registered {person_name} successfully!")
            else:
                print(f"❌ Failed to register {person_name}")
    
    def test_database_status(self):
        """Test database status and statistics"""
        print(f"\n📊 DATABASE STATUS")
        
        # Get database info
        db_info = self.face_db.get_database_info()
        print(f"Collection: {db_info.get('collection_name', 'N/A')}")
        print(f"Total faces: {db_info.get('count', 0)}")
        print(f"Status: {db_info.get('status', 'unknown')}")
        
        # Get all faces (limited)
        faces = self.face_db.get_all_faces(limit=10)
        print(f"\nRecent registrations (max 10):")
        for i, face in enumerate(faces, 1):
            metadata = face['metadata']
            name = metadata.get('name', 'Unknown')
            created_at = metadata.get('created_at', 'N/A')
            print(f"  {i}. {name} (registered: {created_at})")
    
    def run_complete_test(self, register_image=None, register_name=None, test_image=None):
        """
        Run complete test suite
        
        Args:
            register_image (str): Image to register (optional)
            register_name (str): Name for registration (optional)
            test_image (str): Image to test recognition (optional)
        """
        print("🧪 RUNNING COMPLETE TEST SUITE")
        print("="*50)
        
        # Test 1: Database Status
        self.test_database_status()
        
        # Test 2: Registration (if provided)
        if register_image and register_name:
            success = self.test_registration(register_image, register_name)
            if not success:
                print("⚠️  Registration failed, continuing with tests...")
        
        # Test 3: Image Recognition (if provided)
        if test_image:
            results = self.test_recognition_image(test_image)
            if results:
                print(f"✅ Image recognition completed with {len(results)} faces processed")
        
        # Test 4: Real-time Recognition
        print(f"\nStarting real-time test in 3 seconds...")
        time.sleep(3)
        self.test_recognition_realtime(duration_seconds=15)
        
        # Test 5: Final Database Status
        print(f"\nFinal database status:")
        self.test_database_status()
        
        print("✅ Complete test suite finished!")


def main():
    """Main test function"""
    print("🧪 SIMPLE FACE RECOGNITION TEST")
    print("="*40)
    
    # Initialize test
    test = SimpleTest()
    
    # Example usage
    print("\nTest options:")
    print("1. Complete test with sample images")
    print("2. Registration test only")
    print("3. Recognition test only")
    print("4. Real-time test only")
    print("5. Database status only")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Complete test
        register_img = input("Enter path to image for registration (or press Enter to skip): ").strip()
        register_name = input("Enter name for registration (or press Enter to skip): ").strip()
        test_img = input("Enter path to test image (or press Enter to skip): ").strip()
        
        register_img = register_img if register_img else None
        register_name = register_name if register_name else None
        test_img = test_img if test_img else None
        
        test.run_complete_test(register_img, register_name, test_img)
    
    elif choice == "2":
        # Registration only
        image_path = input("Enter path to image: ").strip()
        person_name = input("Enter person name: ").strip()
        test.test_registration(image_path, person_name)
    
    elif choice == "3":
        # Recognition only
        image_path = input("Enter path to test image: ").strip()
        test.test_recognition_image(image_path)
    
    elif choice == "4":
        # Real-time only
        duration = input("Enter test duration in seconds (default 30): ").strip()
        duration = int(duration) if duration else 30
        test.test_recognition_realtime(duration)
    
    elif choice == "5":
        # Database status only
        test.test_database_status()
    
    else:
        print("Invalid choice!")
        return
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()