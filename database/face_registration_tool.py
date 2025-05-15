"""
Face Registration Tool - Standalone CLI
Đăng ký khuôn mặt độc lập
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.chroma_manager import ChromaFaceDB
from app.box_detector import Detector

# Try to import HEIC support
try:
    from PIL import Image
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
    print("HEIC support enabled")
except ImportError:
    HEIC_SUPPORTED = False
    print("HEIC support not available. Install pillow-heif for .heic support")

class FaceRegistrationTool:
    """Standalone face registration tool"""
    
    def __init__(self, db_path="./face_db/chroma_data"):
        """
        Initialize registration tool
        """
        print("Initializing Face Registration Tool...")
        
        # Initialize detector for face detection and embedding
        self.detector = Detector()
        
        # Initialize ChromaDB
        self.face_db = ChromaFaceDB(db_path)
        
        print("Face Registration Tool ready!")
    
    def _convert_heic_to_cv2(self, image_path):
        """
        Convert HEIC image to OpenCV format
        
        Args:
            image_path (str): Path to HEIC image
            
        Returns:
            np.ndarray: OpenCV image in BGR format, or None if failed
        """
        try:
            # Open HEIC with PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL to numpy array
            img_array = np.array(pil_image)
            
            # Convert RGB to BGR (OpenCV format)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
        except Exception as e:
            print(f"Error converting HEIC: {e}")
            return None
    
    def _read_image(self, image_path):
        """
        Read image with support for multiple formats including HEIC
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: OpenCV image in BGR format, or None if failed
        """
        # Check if file is HEIC
        if HEIC_SUPPORTED and image_path.lower().endswith(('.heic', '.HEIC')):
            return self._convert_heic_to_cv2(image_path)
        else:
            # Use OpenCV for standard formats
            return cv2.imread(image_path)
    
    def register_single_face(self, image_path, person_name, person_id=None, metadata=None):
        """
        Register single face from image
        
        Args:
            image_path (str): Path to image file
            person_name (str): Name of the person
            person_id (str, optional): Custom person ID
            metadata (dict, optional): Additional metadata
        
        Returns:
            str: Face ID if successful, None if failed
        """
        print(f"\nRegistering: {person_name}")
        print(f"Image: {image_path}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        # Read image with HEIC support
        image = self._read_image(image_path)
        if image is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        
        print(f"Image size: {image.shape}")
        
        # Detect faces and extract embedding
        start_time = time.time()
        try:
            person_count, face_count, person_boxes, face_boxes = self.detector.process_frame(image)
        except Exception as e:
            print(f"Error during face detection: {e}")
            return None
        
        # Validate face detection results
        if face_count == 0 or not face_boxes or len(face_boxes) == 0:
            print("No faces detected in image")
            return None
        
        # Check if face_count matches actual face_boxes length
        if face_count != len(face_boxes):
            print(f"Warning: face_count ({face_count}) != actual faces detected ({len(face_boxes)})")
            face_count = len(face_boxes)
        
        if face_count > 1:
            print(f"Warning: Multiple faces detected ({face_count}). Using first face.")
        
        # Get embedding from first face - with validation
        face_data = face_boxes[0]
        
        # Validate face_data structure
        if not face_data or len(face_data) < 4:
            print(f"Error: Invalid face data structure: {face_data}")
            return None
        
        coords, conf, emotion, embedding = face_data
        
        # Validate embedding
        if embedding is None:
            print("Failed to extract face embedding")
            return None
        
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        # Validate embedding array
        if embedding.size == 0 or len(embedding.shape) == 0:
            print("Error: Empty or invalid embedding")
            return None
        
        processing_time = time.time() - start_time
        print(f"Face detected (confidence: {conf:.3f})")
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Embedding dimension: {embedding.shape}")
        
        # Prepare metadata
        face_metadata = {
            "name": person_name,
            "image_path": image_path,
            "confidence": float(conf),
            "embedding_dim": embedding.shape[0],
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add custom metadata if provided
        if metadata:
            face_metadata.update(metadata)
        
        # Add person_id if provided
        if person_id:
            face_metadata["person_id"] = person_id
        
        # Add to database
        start_db = time.time()
        try:
            face_id = self.face_db.add_face(embedding, face_metadata, person_id)
        except Exception as e:
            print(f"Error adding to database: {e}")
            return None
        
        db_time = time.time() - start_db
        
        if face_id:
            print(f"Successfully registered face!")
            print(f"Face ID: {face_id}")
            print(f"Database time: {db_time:.3f}s")
            return face_id
        else:
            print("Failed to add face to database")
            return None
    
    def register_folder(self, folder_path, recursive=False, max_images=None):
        """
        Register all faces from a folder
        
        Args:
            folder_path (str): Path to folder containing images
            recursive (bool): Include subdirectories
            max_images (int): Maximum number of images to process
        
        Returns:
            dict: Registration statistics
        """
        print(f"\nProcessing folder: {folder_path}")
        print(f"Recursive: {recursive}")
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder not found at {folder_path}")
            return None
        
        # Find image files with HEIC support
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if HEIC_SUPPORTED:
            image_extensions.extend(['.heic', '.HEIC'])
        
        image_files = []
        
        if recursive:
            for ext in image_extensions:
                image_files.extend(list(Path(folder_path).rglob(f"*{ext}")))
                image_files.extend(list(Path(folder_path).rglob(f"*{ext.upper()}")))
        else:
            for ext in image_extensions:
                image_files.extend(list(Path(folder_path).glob(f"*{ext}")))
                image_files.extend(list(Path(folder_path).glob(f"*{ext.upper()}")))
        
        if not image_files:
            print("No image files found in folder")
            return None
        
        # Apply max_images limit
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"Limiting to {max_images} images")
        
        print(f"Found {len(image_files)} images to process")
        
        # Registration statistics
        stats = {
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_time": 0,
            "results": []
        }
        
        start_total = time.time()
        
        # Process each image
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
            
            # Use filename (without extension) as person name
            person_name = img_path.stem
            
            # Register face
            face_id = self.register_single_face(str(img_path), person_name)
            
            if face_id:
                stats["successful"] += 1
                stats["results"].append({
                    "image": str(img_path),
                    "name": person_name,
                    "face_id": face_id,
                    "status": "success"
                })
            else:
                stats["failed"] += 1
                stats["results"].append({
                    "image": str(img_path),
                    "name": person_name,
                    "face_id": None,
                    "status": "failed"
                })
        
        stats["total_time"] = time.time() - start_total
        
        # Print summary
        print(f"\nREGISTRATION SUMMARY")
        print(f"Total images: {stats['total_images']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {stats['successful']/stats['total_images']*100:.1f}%")
        print(f"Total time: {stats['total_time']:.1f}s")
        print(f"Average per image: {stats['total_time']/stats['total_images']:.2f}s")
        
        return stats
    
    def get_database_info(self):
        """Get database information"""
        return self.face_db.get_database_info()
    
    def register_people_folders(self, base_folder_path, max_images_per_person=None):
        """
        Register multiple people from folder structure
        Each subfolder represents one person, folder name = person name
        
        Args:
            base_folder_path (str): Base folder containing person folders
            max_images_per_person (int): Maximum images per person
        
        Returns:
            dict: Complete registration results
        """
        print(f"\nRegistering multiple people from: {base_folder_path}")
        
        if not os.path.exists(base_folder_path):
            print(f"Base folder not found: {base_folder_path}")
            return None
        
        # Find all person folders (subdirectories)
        person_folders = [d for d in Path(base_folder_path).iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        
        if not person_folders:
            print(f"No person folders found in {base_folder_path}")
            return None
        
        print(f"Found {len(person_folders)} people to register:")
        for folder in person_folders:
            print(f"  - {folder.name}")
        
        # Overall results
        overall_results = {
            "base_folder": base_folder_path,
            "total_people": len(person_folders),
            "people_results": [],
            "total_images": 0,
            "total_successful": 0,
            "total_failed": 0,
            "total_time": 0
        }
        
        start_total = time.time()
        
        # Process each person
        for i, person_folder in enumerate(person_folders, 1):
            person_name = person_folder.name
            print(f"\n[{i}/{len(person_folders)}] " + "="*50)
            print(f"Processing: {person_name}")
            
            # Register this person's folder
            person_result = self.register_person_folder(str(person_folder), person_name, max_images_per_person)
            
            if person_result:
                overall_results["people_results"].append(person_result)
                overall_results["total_images"] += person_result["total_images"]
                overall_results["total_successful"] += person_result["successful"]
                overall_results["total_failed"] += person_result["failed"]
        
        overall_results["total_time"] = time.time() - start_total
        
        # Print overall summary
        print(f"\nOVERALL REGISTRATION SUMMARY")
        print("="*60)
        print(f"People registered: {len(overall_results['people_results'])}/{overall_results['total_people']}")
        print(f"Total images processed: {overall_results['total_images']}")
        print(f"Total successful: {overall_results['total_successful']}")
        print(f"Total failed: {overall_results['total_failed']}")
        print(f"Overall success rate: {overall_results['total_successful']/max(1, overall_results['total_images'])*100:.1f}%")
        print(f"Total time: {overall_results['total_time']:.1f}s")
        print(f"Average per image: {overall_results['total_time']/max(1, overall_results['total_images']):.2f}s")
        
        # Per-person breakdown
        print(f"\nPer-person breakdown:")
        for result in overall_results["people_results"]:
            success_rate = result["successful"]/max(1, result["total_images"])*100
            print(f"  {result['person_name']:20} {result['successful']:3d}/{result['total_images']:3d} ({success_rate:5.1f}%)")
        
        return overall_results
    
    def register_person_folder(self, person_folder_path, person_name, max_images_per_person=None):
        """
        Register all images from a single person's folder
        
        Args:
            person_folder_path (str): Path to person's folder
            person_name (str): Person's name
            max_images_per_person (int): Max images to process per person
        
        Returns:
            dict: Registration results for this person
        """
        print(f"Processing folder: {person_folder_path}")
        
        # Find image files in person's folder with HEIC support
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        if HEIC_SUPPORTED:
            image_extensions.extend(['.heic', '.HEIC'])
        
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(person_folder_path).glob(f"*{ext}")))
            image_files.extend(list(Path(person_folder_path).glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(f"No images found in {person_folder_path}")
            return {
                "person_name": person_name,
                "total_images": 0,
                "successful": 0,
                "failed": 0,
                "face_ids": [],
                "processing_time": 0
            }
        
        # Limit images if specified
        if max_images_per_person and len(image_files) > max_images_per_person:
            image_files = image_files[:max_images_per_person]
            print(f"Limited to {max_images_per_person} images")
        
        print(f"Found {len(image_files)} images for {person_name}")
        
        # Registration results for this person
        person_results = {
            "person_name": person_name,
            "total_images": len(image_files),
            "successful": 0,
            "failed": 0,
            "face_ids": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        # Process each image for this person
        for i, img_path in enumerate(image_files, 1):
            print(f"  [{i}/{len(image_files)}] Processing: {img_path.name}")
            
            # Register face with person name
            face_id = self.register_single_face(
                str(img_path), 
                person_name,
                metadata={
                    "source_folder": person_folder_path,
                    "image_file": img_path.name,
                    "person_image_number": i
                }
            )
            
            if face_id:
                person_results["successful"] += 1
                person_results["face_ids"].append(face_id)
                print(f"    Success")
            else:
                person_results["failed"] += 1
                print(f"    Failed")
        
        person_results["processing_time"] = time.time() - start_time
        
        # Print person summary
        print(f"\n{person_name} Summary:")
        print(f"  Success: {person_results['successful']}")
        print(f"  Failed: {person_results['failed']}")
        print(f"  Success rate: {person_results['successful']/max(1, person_results['total_images'])*100:.1f}%")
        print(f"  Time: {person_results['processing_time']:.1f}s")
        
        return person_results
    
    def reset_database(self):
        """Reset the face database"""
        import shutil
        
        print("Resetting database...")
        
        # Close current connection
        del self.face_db
        
        # Delete database folder
        db_path = "./face_db/chroma_data"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print("Database folder deleted")
        
        # Reinitialize database
        self.face_db = ChromaFaceDB(db_path)
        print("Database reset complete!")
    
    def get_database_stats(self):
        """Get detailed database statistics"""
        print(f"\nDATABASE STATISTICS")
        print("="*40)
        
        # Basic info
        db_info = self.get_database_info()
        print(f"Total faces: {db_info.get('count', 0)}")
        print(f"Collection: {db_info.get('collection_name', 'N/A')}")
        
        # Get all faces for analysis
        all_faces = self.face_db.get_all_faces(limit=1000)
        
        if all_faces:
            # Analyze by person
            people_stats = {}
            for face in all_faces:
                metadata = face['metadata']
                name = metadata.get('name', 'Unknown')
                
                if name not in people_stats:
                    people_stats[name] = 0
                people_stats[name] += 1
            
            print(f"\nRegistered people: {len(people_stats)}")
            print("Face count per person:")
            for name, count in sorted(people_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name:20} {count:3d} faces")
        
        return db_info
    
    def search_face(self, image_path, top_k=5, threshold=0.7):
        """
        Search for similar faces (for testing)
        """
        print(f"\nSearching for similar faces...")
        
        # Read and process image with HEIC support
        image = self._read_image(image_path)
        if image is None:
            print(f"Cannot read image {image_path}")
            return None
        
        # Get embedding
        _, _, _, face_boxes = self.detector.process_frame(image)
        if not face_boxes:
            print("No faces detected")
            return None
        
        embedding = face_boxes[0][3]
        if embedding is None:
            print("Failed to extract embedding")
            return None
        
        # Search in database
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        results = self.face_db.search_faces(embedding, top_k, threshold)
        
        print(f"Found {len(results)} similar faces:")
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            print(f"{i}. {metadata.get('name', 'Unknown')} (similarity: {result['similarity']:.3f})")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Face Registration Tool')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--name', type=str, help='Person name (required with --image)')
    parser.add_argument('--person_id', type=str, help='Custom person ID')
    parser.add_argument('--folder', type=str, help='Path to folder of images (same person)')
    parser.add_argument('--people_folder', type=str, help='Path to folder containing person folders (each subfolder = one person)')
    parser.add_argument('--recursive', action='store_true', help='Process subfolders recursively')
    parser.add_argument('--max_images', type=int, help='Maximum number of images to process')
    parser.add_argument('--max_per_person', type=int, help='Maximum images per person (for people_folder)')
    parser.add_argument('--db_path', type=str, default='./face_db/chroma_data', help='Database path')
    parser.add_argument('--search', type=str, help='Search for similar faces')
    parser.add_argument('--info', action='store_true', help='Show database info')
    parser.add_argument('--stats', action='store_true', help='Show detailed database statistics')
    parser.add_argument('--reset', action='store_true', help='Reset database')
    
    args = parser.parse_args()
    
    # Initialize registration tool
    tool = FaceRegistrationTool(args.db_path)
    
    # Reset database if requested
    if args.reset:
        tool.reset_database()
        return
    
    # Show database info
    if args.info:
        info = tool.get_database_info()
        print(f"\nDatabase Info:")
        print(f"Collection: {info.get('collection_name', 'N/A')}")
        print(f"Total faces: {info.get('count', 0)}")
        print(f"Status: {info.get('status', 'unknown')}")
    
    # Show detailed stats
    elif args.stats:
        tool.get_database_stats()
    
    # Single image registration
    elif args.image:
        if not args.name:
            print("Error: --name is required when using --image")
            sys.exit(1)
        
        tool.register_single_face(args.image, args.name, args.person_id)
    
    # Folder registration (same person)
    elif args.folder:
        # For folder registration, use folder name as person name if no specific name provided
        person_name = args.name if args.name else os.path.basename(args.folder)
        print(f"Registering all images as: {person_name}")
        tool.register_folder(args.folder, args.recursive, args.max_images)
    
    # People folder registration (each subfolder = one person)
    elif args.people_folder:
        tool.register_people_folders(args.people_folder, args.max_per_person)
    
    # Search for similar faces
    elif args.search:
        tool.search_face(args.search)
    
    else:
        print("Error: Specify an action")
        print("\nExamples:")
        print("  # Reset database")
        print("  python database/face_registration_tool.py --reset")
        print("  ")
        print("  # Register single image")
        print("  python database/face_registration_tool.py --image face.jpg --name 'John Doe'")
        print("  ")
        print("  # Register folder (same person)")
        print("  python database/face_registration_tool.py --folder ./john_faces/ --name 'John Doe'")
        print("  ")
        print("  # Register multiple people (each subfolder = one person)")
        print("  python database/face_registration_tool.py --people_folder ./faces/")
        print("  ")
        print("  # Show database stats")
        print("  python database/face_registration_tool.py --stats")
        print("Use --help for all options")
        sys.exit(1)


if __name__ == "__main__":
    main()