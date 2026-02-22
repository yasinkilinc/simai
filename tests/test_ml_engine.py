import unittest
import os
import shutil
import json
import numpy as np
from src.ml_engine import MLEngine

class TestMLEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_data"
        self.model_path = "test_model.joblib"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy annotations
        self.create_dummy_annotation("img1", "Oval", 0.8, 0.9)
        self.create_dummy_annotation("img2", "Kare", 0.9, 0.95)
        self.create_dummy_annotation("img3", "Oval", 0.79, 0.88)
        self.create_dummy_annotation("img4", "Kare", 0.92, 0.92)
        self.create_dummy_annotation("img5", "Oval", 0.81, 0.89)
        self.create_dummy_annotation("img6", "Kare", 0.91, 0.93)
        self.create_dummy_annotation("img7", "Oval", 0.82, 0.87)
        self.create_dummy_annotation("img8", "Kare", 0.93, 0.94)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            
    def create_dummy_annotation(self, name, shape, wh_ratio, jaw_ratio):
        data = {
            "face_shape": {"shape": shape},
            "metrics": {
                "face_wh_ratio": wh_ratio,
                "jaw_face_width_ratio": jaw_ratio,
                "forehead_jaw_ratio": 1.0,
                "eye_spacing_ratio": 1.0,
                "eye_size_ratio": 0.25,
                "nose_ratio": 0.7,
                "nose_bridge_ratio": 0.3,
                "lip_ratio": 1.0,
                "eyebrow_arch": 0.0,
                "chin_angle": 120,
                "cheek_jaw_ratio": 1.0,
                "forehead_h_w_ratio": 0.5
            }
        }
        with open(os.path.join(self.test_dir, f"{name}_annotation.json"), 'w') as f:
            json.dump(data, f)

    def test_train_and_predict(self):
        engine = MLEngine()
        
        # Train
        success = engine.train(self.test_dir)
        self.assertTrue(success)
        self.assertIn('face_shape', engine.models)
        
        # Save
        engine.save_model(self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        
        # Load new instance
        engine2 = MLEngine(self.model_path)
        self.assertIn('face_shape', engine2.models)
        
        # Predict
        features = {
            "face_wh_ratio": 0.8,
            "jaw_face_width_ratio": 0.9,
            "forehead_jaw_ratio": 1.0,
            "eye_spacing_ratio": 1.0,
            "eye_size_ratio": 0.25,
            "nose_ratio": 0.7,
            "nose_bridge_ratio": 0.3,
            "lip_ratio": 1.0,
            "eyebrow_arch": 0.0,
            "chin_angle": 120,
            "cheek_jaw_ratio": 1.0,
            "forehead_h_w_ratio": 0.5
        }
        result = engine2.predict(features)
        self.assertIsNotNone(result)
        self.assertIn('face_shape', result)
        print(f"Prediction: {result}")

if __name__ == '__main__':
    unittest.main()
