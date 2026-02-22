import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml_engine import MLEngine

def main():
    parser = argparse.ArgumentParser(description='Train Face Personality Model')
    parser.add_argument('--data_dir', type=str, default='dataset/annotations', help='Path to annotations directory')
    parser.add_argument('--output', type=str, default='models/face_personality_rf.joblib', help='Path to save model')
    
    args = parser.parse_args()
    
    print(f"Starting training with data from {args.data_dir}...")
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    engine = MLEngine()
    success = engine.train(args.data_dir)
    
    if success:
        engine.save_model(args.output)
        print("Training completed successfully.")
    else:
        print("Training failed.")

if __name__ == "__main__":
    main()
