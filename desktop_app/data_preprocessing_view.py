from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QSpinBox, QCheckBox
)
from PySide6.QtCore import QThread, Signal
import os
import cv2
import csv
from datetime import datetime
import time

class PreprocessWorker(QThread):
    """Worker thread for preprocessing images"""
    log_message = Signal(str)
    progress_update = Signal(int)
    finished = Signal()
    
    def __init__(self, raw_dir, processed_dir, target_size=1024, face_align=False):
        super().__init__()
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.target_size = target_size
        self.face_align = face_align
        self.is_running = True
        
        # Load Mediapipe Face Detection for bounding box
        self.face_detection = None
        self.face_mesh = None
        if self.face_align:
            try:
                import mediapipe as mp
                # Face detection for bounding box
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 1 for full range, 0 for short range
                    min_detection_confidence=0.5
                )
                # Face mesh for eye alignment  
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            except ImportError:
                self.log_message.emit("⚠️ Mediapipe kurulu değil! Yüz hizalama devre dışı.")
                self.face_align = False
            except Exception as e:
                self.log_message.emit(f"⚠️ Mediapipe başlatılamadı: {e}")
                self.face_align = False
    
    def align_face(self, img):
        """Advanced face alignment using Mediapipe with bounding box detection"""
        if self.face_detection is None or self.face_mesh is None:
            return img
        
        import numpy as np
        
        h, w = img.shape[:2]
        
        # Convert to RGB for Mediapipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Detect face bounding box
        detection_results = self.face_detection.process(rgb_img)
        if not detection_results.detections:
            return img  # No face detected
        
        detection = detection_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bbox_w = int(bbox.width * w)
        bbox_h = int(bbox.height * h)
        
        # 2. Add padding (100% top for full hair coverage, 50% bottom for neck, 40% sides)
        padding_top = int(bbox_h * 1.00)  # Double the face height for top
        padding_bottom = int(bbox_h * 0.50)
        padding_side = int(bbox_w * 0.40)
        
        # Calculate expanded bounding box
        x1 = max(0, x - padding_side)
        y1 = max(0, y - padding_top)
        x2 = min(w, x + bbox_w + padding_side)
        y2 = min(h, y + bbox_h + padding_bottom)
        
        # 3. Get eye positions for rotation
        mesh_results = self.face_mesh.process(rgb_img)
        if not mesh_results.multi_face_landmarks:
            # No landmarks, just crop and resize without rotation
            cropped = img[y1:y2, x1:x2]
            return cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_LANCZOS4)
        
        landmarks = mesh_results.multi_face_landmarks[0]
        
        # Get eye centers
        left_eye_outer = landmarks.landmark[33]
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]
        right_eye_outer = landmarks.landmark[263]
        
        left_eye = np.array([
            int((left_eye_outer.x + left_eye_inner.x) / 2 * w),
            int((left_eye_outer.y + left_eye_inner.y) / 2 * h)
        ])
        right_eye = np.array([
            int((right_eye_inner.x + right_eye_outer.x) / 2 * w),
            int((right_eye_inner.y + right_eye_outer.y) / 2 * h)
        ])
        
        # 4. Calculate rotation angle to align eyes horizontally
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center for rotation
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # 5. Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # 6. Crop the padded region
        cropped = rotated[y1:y2, x1:x2]
        
        # 7. Resize to target size
        aligned = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_LANCZOS4)
        
        return aligned
        
    def run(self):
        self.log_message.emit(f"🚀 Veri işleme başlatılıyor...")
        self.log_message.emit(f"📂 Kaynak: {self.raw_dir}")
        self.log_message.emit(f"📂 Hedef: {self.processed_dir}")
        self.log_message.emit(f"📐 Hedef boyut: {self.target_size}x{self.target_size}")
        
        # Create train/val/test directories
        train_dir = os.path.join(self.processed_dir, "train")
        val_dir = os.path.join(self.processed_dir, "val")
        test_dir = os.path.join(self.processed_dir, "test")
        
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Collect all images
        all_images = []
        for source_folder in os.listdir(self.raw_dir):
            source_path = os.path.join(self.raw_dir, source_folder)
            if not os.path.isdir(source_path):
                continue
                
            for filename in os.listdir(source_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append((source_folder, filename, source_path))
        
        total = len(all_images)
        self.log_message.emit(f"📊 Toplam {total} resim bulundu.")
        
        # Shuffle for random split
        import random
        random.shuffle(all_images)
        
        # Split: 70% train, 15% val, 15% test
        train_count = int(total * 0.70)
        val_count = int(total * 0.15)
        
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:]
        
        self.log_message.emit(f"📋 Bölünme: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        # Metadata CSV
        dataset_dir = os.path.dirname(self.processed_dir)
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        metadata_rows = []
        
        # Process all images
        for idx, (source, filename, source_path) in enumerate(all_images):
            if not self.is_running:
                break
                
            try:
                # Determine target directory
                if idx < train_count:
                    target_dir = train_dir
                    split = "train"
                elif idx < train_count + val_count:
                    target_dir = val_dir
                    split = "val"
                else:
                    target_dir = test_dir
                    split = "test"
                
                # Read image
                img_path = os.path.join(source_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    self.log_message.emit(f"⚠️ Okunamadı: {filename}")
                    continue
                
                original_h, original_w = img.shape[:2]
                
                # Face alignment if enabled (outputs target_size directly)
                if self.face_align:
                    resized = self.align_face(img)
                else:
                    # Just resize without alignment
                    resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LANCZOS4)
                
                # Save to processed (use original filename to avoid duplicate source prefix)
                output_path = os.path.join(target_dir, filename)
                cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Metadata
                metadata_rows.append({
                    'filename': filename,
                    'source': source,
                    'split': split,
                    'original_size': f"{original_w}x{original_h}",
                    'processed_size': f"{self.target_size}x{self.target_size}",
                    'format': 'JPEG',
                    'timestamp': datetime.now().isoformat()
                })
                
                # Progress
                progress = int(((idx + 1) / total) * 100)
                self.progress_update.emit(progress)
                
            except Exception as e:
                self.log_message.emit(f"❌ Hata ({filename}): {str(e)}")
        
        # Write metadata
        if metadata_rows:
            with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'source', 'split', 'original_size', 'processed_size', 'format', 'timestamp'])
                writer.writeheader()
                writer.writerows(metadata_rows)
            
            self.log_message.emit(f"✅ metadata.csv oluşturuldu: {metadata_path}")
        
        self.log_message.emit(f"\n🎉 İşlem tamamlandı! {len(metadata_rows)}/{total} resim işlendi.")
        self.finished.emit()
    
    def stop(self):
        self.is_running = False

class DataPreprocessingView(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("🔧 Veri Standartları ve Önişleme")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(header)
        
        # Settings Group
        settings_group = QGroupBox("Ayarlar")
        settings_layout = QVBoxLayout()
        
        # Resolution
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Hedef Çözünürlük:"))
        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(256, 4096)
        self.spin_resolution.setSingleStep(256)
        self.spin_resolution.setValue(1024)
        self.spin_resolution.setSuffix(" px")
        resolution_layout.addWidget(self.spin_resolution)
        resolution_layout.addStretch()
        settings_layout.addLayout(resolution_layout)
        
        # Face alignment
        self.chk_face_align = QCheckBox("Yüz Hizalama (Gözler yatay düzlemde)")
        self.chk_face_align.setChecked(True)
        settings_layout.addWidget(self.chk_face_align)
        
        settings_group.setLayout(settings_layout)
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #1e1e2e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout.addWidget(settings_group)
        
        # Info
        info = QLabel("""
📋 <b>Yapılacaklar:</b><br>
• <code>dataset/raw/</code> klasöründeki tüm resimler okunacak<br>
• Her resim 1024x1024 boyutuna getirilecek<br>
• <code>dataset/processed/</code> klasörüne kaydedilecek<br>
• <code>dataset/metadata.csv</code> dosyası oluşturulacak
        """)
        info.setWordWrap(True)
        info.setStyleSheet("color: #a6adc8; background-color: #1e1e2e; padding: 10px; border-radius: 6px;")
        layout.addWidget(info)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e2e;
                color: #1e1e2e;
            }
            QProgressBar::chunk {
                background-color: #a6e3a1;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        layout.addWidget(self.log_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 İşlemeyi Başlat")
        self.btn_start.clicked.connect(self.start_preprocessing)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; 
                color: #1e1e2e; 
                font-weight: bold; 
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        
        self.btn_stop = QPushButton("🛑 Durdur")
        self.btn_stop.clicked.connect(self.stop_preprocessing)
        self.btn_stop.setEnabled(False)  # Disabled at startup
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8; 
                color: #1e1e2e; 
                font-weight: bold; 
                padding: 10px;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
    def start_preprocessing(self):
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_dir = os.path.join(base_dir, "dataset", "raw")
        processed_dir = os.path.join(base_dir, "dataset", "processed")
        metadata_path = os.path.join(base_dir, "dataset", "metadata.csv")
        
        if not os.path.exists(raw_dir):
            self.log("❌ 'dataset/raw' klasörü bulunamadı!")
            return
        
        # Cleanup processed folder
        import shutil
        if os.path.exists(processed_dir):
            self.log("🗑️ Önceki işlenmiş veriler temizleniyor...")
            shutil.rmtree(processed_dir)
        
        # Cleanup metadata.csv
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            self.log("🗑️ Önceki metadata.csv silindi.")
        
        resolution = self.spin_resolution.value()
        face_align = self.chk_face_align.isChecked()
        
        self.worker = PreprocessWorker(raw_dir, processed_dir, resolution, face_align)
        self.worker.log_message.connect(self.log)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        
        self.worker.start()
        
        # Button logic
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
    def stop_preprocessing(self):
        if self.worker:
            self.worker.stop()
            self.log("🛑 İşlem durduruluyor...")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            
    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
    def log(self, msg):
        self.log_text.append(msg)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
