import cv2
import numpy as np
import sys
import os
from PySide6.QtGui import QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import QApplication

# Add src to path
sys.path.insert(0, os.getcwd())

from src.visualization import Visualizer
from annotation_engine import AutoAnnotator

def test_visualization():
    app = QApplication(sys.argv)
    
    # 1. Load Image
    image_path = "/Users/yasinkilinc/.gemini/antigravity/brain/23b1e14a-6295-4d5b-ab3d-7b3027198910/uploaded_image_1763896549248.png"
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    print(f"Original Image Shape: {img.shape}")
    
    # 2. Annotate
    annotator = AutoAnnotator()
    landmarks = annotator.get_landmarks(img)
    if not landmarks:
        print("No landmarks found!")
        return
        
    print("Landmarks found.")
    
    # 3. Simulate MainWindow conversion
    # Convert BGR to RGB for QImage
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(q_img)
    
    print(f"Pixmap size: {pixmap.width()}x{pixmap.height()}")
    
    # 4. Draw
    vis_pixmap = Visualizer.draw_measurements(pixmap, landmarks, annotator)
    
    # 5. Save Result
    res_qimg = vis_pixmap.toImage()
    res_qimg.save("test_vis_result.png")
    print("Saved test_vis_result.png")
    
    # Check if modified
    # We can't easily check pixels without reading back, but saving is enough for visual inspection.

if __name__ == "__main__":
    test_visualization()
