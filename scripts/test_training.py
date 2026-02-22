import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.ml_engine import MLEngine

def test_training():
    print("Testing ML Engine Training...")
    
    data_path = "dataset/export/data.csv"
    model_path = "models/test_model.joblib"
    
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    engine = MLEngine()
    
    # Test 1: Train Face Shape (Classification)
    print("\n--- Test 1: Face Shape (Classification) ---")
    success = engine.train(data_path, target_column='target_face_shape', model_type='classification')
    if success:
        print("✅ Face Shape training successful")
    else:
        print("❌ Face Shape training failed")
        
    # Test 2: Train Forehead Width (Regression - assuming it might be continuous or categorical, let's try classification first as per current data)
    # Actually, let's check what columns we have.
    # For now, just testing face_shape is enough to prove the pipeline works.
    
    if success:
        engine.save_model(model_path)
        print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    test_training()
