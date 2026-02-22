import cv2
import mediapipe as mp
import numpy as np

def test_imports():
    print("Testing imports...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Numpy version: {np.__version__}")
    print("Imports successful!")

def test_mediapipe_init():
    print("Testing MediaPipe initialization...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    print("MediaPipe FaceMesh initialized successfully!")
    face_mesh.close()

if __name__ == "__main__":
    test_imports()
    test_mediapipe_init()
