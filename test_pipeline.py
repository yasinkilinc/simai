import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from reconstruction import FaceReconstructor
from features import FaceFeatures
from interpreter import PhysiognomyInterpreter

def test_pipeline():
    print("Testing pipeline with dummy data...")
    
    # Create a black image
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        reconstructor = FaceReconstructor(static_image_mode=True)
        interpreter = PhysiognomyInterpreter()
        
        print("Processing dummy frame...")
        landmarks = reconstructor.process_frame(dummy_frame)
        
        if landmarks is None:
            print("Correctly detected no face in black image.")
        else:
            # If a face was somehow detected (unlikely in black image), test the full pipeline
            points_3d = reconstructor.get_3d_points(landmarks, dummy_frame.shape)
            features = FaceFeatures(points_3d)
            report = interpreter.interpret(features)
            print("Report generated successfully.")
            
        print("Pipeline test passed!")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
