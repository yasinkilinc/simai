import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from processor import InputProcessor
from reconstruction import FaceReconstructor
from features import FaceFeatures
from interpreter import PhysiognomyInterpreter

def test_real_image():
    print("Testing with real image...")
    image_path = "sample_face.jpg"
    
    if not os.path.exists(image_path):
        print("Image not found.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image.")
        return

    print(f"Image loaded: {frame.shape}")

    reconstructor = FaceReconstructor()
    landmarks = reconstructor.process_frame(frame)
    
    if landmarks:
        print("Face detected!")
        points_3d = reconstructor.get_3d_points(landmarks, frame.shape)
        print(f"3D Points: {points_3d.shape}")
        
        features = FaceFeatures(points_3d, frame)
        print("Features extracted.")
        
        interpreter = PhysiognomyInterpreter()
        report = interpreter.interpret(features)
        print("Interpretation done.")
        print(f"Shape: {report['face_shape']}")
    else:
        print("No face detected.")

if __name__ == "__main__":
    test_real_image()
