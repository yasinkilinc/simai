from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import os
import json
from datetime import datetime

class FeedbackDialog(QDialog):
    def __init__(self, parent=None, image=None, current_prediction=None, image_path=None):
        super().__init__(parent)
        self.image = image
        self.current_prediction = current_prediction
        self.image_path = image_path
        self.setWindowTitle("AI Geri Bildirimi")
        self.setMinimumSize(500, 600)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #cdd6f4; }
            QLabel { color: #cdd6f4; }
            QComboBox { 
                background-color: #181825; 
                color: #cdd6f4; 
                padding: 8px; 
                border: 1px solid #313244; 
                border-radius: 4px;
            }
            QTextEdit { 
                background-color: #181825; 
                color: #cdd6f4; 
                border: 1px solid #313244; 
                border-radius: 4px;
            }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Image Preview
        if self.image is not None:
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setFixedHeight(200)
            
            # Convert CV2 image to QPixmap
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            img_label.setPixmap(pixmap.scaled(300, 200, Qt.KeepAspectRatio))
            layout.addWidget(img_label)
            
        # Current Prediction
        pred_text = self.current_prediction.get('face_shape', 'Bilinmiyor') if self.current_prediction else 'Bilinmiyor'
        layout.addWidget(QLabel(f"AI Tahmini: {pred_text}"))
        
        layout.addSpacing(20)
        
        # Correction Form
        layout.addWidget(QLabel("Doğru Yüz Şekli:"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItems([
            "Oval", "Kare", "Yuvarlak", "Dikdörtgen", 
            "Üçgen", "Elmas", "Kalp", "Uzun"
        ])
        
        # Set current if matches
        if self.current_prediction and 'face_shape' in self.current_prediction:
            index = self.shape_combo.findText(self.current_prediction['face_shape'])
            if index >= 0:
                self.shape_combo.setCurrentIndex(index)
                
        layout.addWidget(self.shape_combo)
        
        layout.addWidget(QLabel("Notlar (Opsiyonel):"))
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        layout.addWidget(self.notes_edit)
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("İptal")
        btn_cancel.setStyleSheet("background-color: #45475a; color: #cdd6f4;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_save = QPushButton("Kaydet ve Öğret")
        btn_save.clicked.connect(self.save_feedback)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        layout.addLayout(btn_layout)
        
    def save_feedback(self):
        """Save the correction as a new annotation"""
        if not self.image_path:
            QMessageBox.warning(self, "Hata", "Görüntü yolu bulunamadı, kaydedilemiyor.")
            return
            
        try:
            # Create annotation data
            annotation = {
                "face_shape": {"shape": self.shape_combo.currentText()},
                "notes": self.notes_edit.toPlainText(),
                "timestamp": datetime.now().isoformat(),
                "corrected_by_user": True,
                "original_prediction": self.current_prediction
            }
            
            # Save to dataset/annotations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            annotations_dir = os.path.join(base_dir, "dataset", "annotations")
            os.makedirs(annotations_dir, exist_ok=True)
            
            filename = os.path.basename(self.image_path)
            annotation_filename = os.path.splitext(filename)[0] + '_annotation.json'
            save_path = os.path.join(annotations_dir, annotation_filename)
            
            # If we have features/metrics from the analysis, we should save them too!
            # But we don't have them passed here easily. 
            # Ideally, we should pass the full 'report' or 'features' object.
            # For now, we just save the label. The training script might need to re-extract features.
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
                
            QMessageBox.information(self, "Başarılı", "Geri bildirim kaydedildi! Bir sonraki eğitimde dikkate alınacak.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kaydetme hatası: {str(e)}")
