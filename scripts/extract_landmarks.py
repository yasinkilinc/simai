import os
import cv2
import json
import mediapipe as mp
import argparse
from tqdm import tqdm
import numpy as np

def extract_landmarks(image_dir, output_file):
    """
    Extracts face landmarks from images in a directory using MediaPipe Face Mesh.
    Saves the result as a JSON file: {image_name: [[x, y, z], ...]}
    """
    
    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize Face Mesh
    # static_image_mode=True for processing independent images
    # refine_landmarks=True for detailed eyes and lips
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        results_data = {}
        failed_images = []
        
        # Get list of images
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images in {image_dir}")
        
        for img_name in tqdm(image_files, desc="Extracting Landmarks"):
            img_path = os.path.join(image_dir, img_name)
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {img_name}")
                failed_images.append(img_name)
                continue
                
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                # Get the first face
                face_landmarks = results.multi_face_landmarks[0]
                
                # Convert to list of [x, y, z]
                # Note: x, y are normalized [0.0, 1.0]
                landmarks_list = []
                for lm in face_landmarks.landmark:
                    landmarks_list.append([lm.x, lm.y, lm.z])
                    
                results_data[img_name] = landmarks_list
            else:
                # No face detected
                failed_images.append(img_name)
        
        # Save results
        print(f"Saving landmarks for {len(results_data)} images to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f)
            
        if failed_images:
            print(f"Failed to detect face in {len(failed_images)} images.")
            # Save failed list
            failed_log = output_file.replace('.json', '_failed.txt')
            with open(failed_log, 'w') as f:
                for img in failed_images:
                    f.write(f"{img}\n")
            print(f"Failed images list saved to {failed_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract MediaPipe Landmarks')
    parser.add_argument('--img_dir', type=str, default='dataset/export/images', help='Directory containing images')
    parser.add_argument('--output', type=str, default='dataset/export/landmarks.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory not found at {args.img_dir}")
    else:
        extract_landmarks(args.img_dir, args.output)
