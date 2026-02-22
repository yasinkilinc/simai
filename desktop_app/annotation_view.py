from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGroupBox, QFormLayout, QScrollArea, QFrame,
    QCheckBox, QGridLayout, QSizePolicy, QProgressDialog, QMessageBox, QApplication
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PySide6.QtCore import Qt, QPoint, QBuffer, QIODevice, QTimer, QSettings
import cv2
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation_engine import AutoAnnotator, AnnotationManager
from src.visualization import Visualizer

class AnnotationView(QWidget):
    """Physiognomic annotation interface"""
    
    def __init__(self):
        super().__init__()
        self.annotator = AutoAnnotator()
        self.manager = AnnotationManager()
        
        # Initialize IDs
        self.manager.refresh_ids()
        self.current_index = 0
        
        self.current_image_data = None # {id, image_name, image, ...}
        self.current_clean_image = None # numpy array
        self.current_annotated_image = None # QPixmap
        
        self.setup_ui()
        
        # Load last index from settings
        self.settings = QSettings("FizyonomiAI", "AnnotationView")
        last_index = self.settings.value("last_index", 0, type=int)
        
        # Validate index
        total_count = self.manager.get_total_count()
        if last_index >= total_count:
            last_index = 0
            
        self.load_image_at_index(last_index)
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left Panel: Image Display + Info
        left_panel = QVBoxLayout()
        
        # Header Layout (Title + Info)
        header_layout = QHBoxLayout()
        
        header = QLabel("🏷️ Fizyonomik Etiketleme")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Batch Auto-Annotate Button
        self.btn_batch_auto = QPushButton("⚡ Toplu Otomatik Etiketle")
        self.btn_batch_auto.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold; padding: 8px;")
        self.btn_batch_auto.clicked.connect(self.batch_auto_annotate)
        header_layout.addWidget(self.btn_batch_auto)
        
        header_layout.addSpacing(15)
        
        # Info in Header
        self.lbl_image_name = QLabel("-")
        self.lbl_image_size = QLabel("-")
        
        info_style = "color: #a6adc8; font-size: 12px; font-weight: bold;"
        val_style = "color: #cdd6f4; font-size: 12px;"
        
        header_layout.addWidget(QLabel("Dosya:", styleSheet=info_style))
        self.lbl_image_name.setStyleSheet(val_style)
        header_layout.addWidget(self.lbl_image_name)
        
        header_layout.addSpacing(15)
        
        header_layout.addWidget(QLabel("Boyut:", styleSheet=info_style))
        self.lbl_image_size.setStyleSheet(val_style)
        header_layout.addWidget(self.lbl_image_size)
        
        left_panel.addLayout(header_layout)
        
        # Image display
        self.image_label = QLabel("Fotoğraf yükleniyor...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 100)
        self.image_label.setStyleSheet("background-color: #11111b; border: 1px solid #45475a; border-radius: 6px;")
        self.image_label.setScaledContents(False)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        left_panel.addWidget(self.image_label, 1)
        
        # Bottom Controls (Toggle + Face Shape)

        bottom_controls = QHBoxLayout()
        
        # Toggle Button
        self.btn_toggle_annotations = QPushButton("👁️ Etiketleri Gizle")
        self.btn_toggle_annotations.setCheckable(True)
        self.btn_toggle_annotations.setChecked(True) # Default Shown
        self.btn_toggle_annotations.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 8px;")
        self.btn_toggle_annotations.clicked.connect(self.toggle_annotations)
        bottom_controls.addWidget(self.btn_toggle_annotations)
        
        # Face Shape
        bottom_controls.addSpacing(20)
        bottom_controls.addWidget(QLabel("Yüz Şekli:"))
        self.combo_face_shape = QComboBox()
        self.combo_face_shape.addItems(["Belirsiz", "Oval", "Yuvarlak", "Kare", "Dikdörtgen", "Üçgen"])
        self.combo_face_shape.setStyleSheet("""
            QComboBox { background-color: #313244; color: #cdd6f4; padding: 5px; border-radius: 4px; min-width: 120px; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #cdd6f4; margin-right: 5px; }
        """)
        bottom_controls.addWidget(self.combo_face_shape)
        
        bottom_controls.addStretch()
        left_panel.addLayout(bottom_controls)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Önceki")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_prev.setStyleSheet("background-color: #45475a; color: #cdd6f4; padding: 10px; font-weight: bold;")
        
        self.label_progress = QLabel("0/0")
        self.label_progress.setAlignment(Qt.AlignCenter)
        self.label_progress.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6e3a1;")
        
        self.btn_next = QPushButton("Sonraki ▶")
        self.btn_next.clicked.connect(self.next_image)
        self.btn_next.setStyleSheet("background-color: #45475a; color: #cdd6f4; padding: 10px; font-weight: bold;")
        
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.label_progress, 1)
        nav_layout.addWidget(self.btn_next)
        left_panel.addLayout(nav_layout)
        
        # Right Panel (Controls)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        right_panel.setAlignment(Qt.AlignTop)
        
        # Annotation Controls (Scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        self.controls_layout.setSpacing(15)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create Combos in Groups
        self.create_combos()
        
        self.controls_layout.addStretch()
        scroll.setWidget(controls_widget)
        
        right_panel.addWidget(scroll, stretch=1)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_export = QPushButton("📤 Veri Setini Dışa Aktar (CSV)")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.clicked.connect(self.export_dataset_ui)
        self.btn_export.setStyleSheet("background-color: #f9e2af; color: #1e1e2e; font-weight: bold;")
        
        # Save button removed - auto-save on navigation
        # self.btn_save = QPushButton("💾 Kaydet")
        # self.btn_save.setMinimumHeight(50)
        # self.btn_save.clicked.connect(self.save_annotation)
        # self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; font-size: 14px;")
        
        btn_layout.addWidget(self.btn_export)
        # btn_layout.addWidget(self.btn_save)
        right_panel.addLayout(btn_layout)
        
        # Add panels to main layout
        layout.addLayout(left_panel, 2)
        layout.addLayout(right_panel, 1)

    def create_combos(self):
        """Create dropdowns for annotation in grouped layout"""
        
        def create_group(title, items):
            group = QGroupBox(title)
            group.setStyleSheet("""
                QGroupBox { font-weight: bold; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; } 
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            """)
            layout = QGridLayout(group)
            layout.setSpacing(10)
            
            row = 0
            col = 0
            for label_text, options, member_name in items:
                container = QWidget()
                v_layout = QVBoxLayout(container)
                v_layout.setContentsMargins(0, 0, 0, 0)
                v_layout.setSpacing(2)
                
                lbl = QLabel(label_text)
                lbl.setStyleSheet("color: #a6adc8; font-size: 11px;")
                v_layout.addWidget(lbl)
                
                combo = QComboBox()
                combo.addItems(options)
                combo.setStyleSheet("""
                    QComboBox { background-color: #313244; color: #cdd6f4; padding: 4px; border-radius: 4px; font-size: 12px; }
                    QComboBox::drop-down { border: none; }
                    QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 4px solid #cdd6f4; margin-right: 4px; }
                """)
                setattr(self, member_name, combo)
                v_layout.addWidget(combo)
                
                layout.addWidget(container, row, col)
                col += 1
                if col > 1: # 2 columns
                    col = 0
                    row += 1
            
            self.controls_layout.addWidget(group)

        # Forehead
        create_group("Alın Yapısı", [
            ("Genişlik", ["Normal", "Geniş", "Dar"], "combo_forehead_width"),
            ("Yükseklik", ["Normal", "Yüksek", "Kısa"], "combo_forehead_height"),
            ("Eğim", ["Düz", "Eğimli", "Yuvarlak"], "combo_forehead_slope")
        ])
        
        # Eyes
        create_group("Göz Yapısı", [
            ("Boyut", ["Normal", "Büyük", "Küçük"], "combo_eyes_size"),
            ("Eğim", ["Düz", "Çekik", "Düşük"], "combo_eyes_slant"),
            ("Aralık", ["Normal", "Ayrık", "Bitişik"], "combo_eyes_spacing"),
            ("Çukur", ["Normal", "Çukur", "Çıkık"], "combo_eyes_depth")
        ])
        
        # Nose
        create_group("Burun Yapısı", [
            ("Boy", ["Normal", "Uzun", "Kısa"], "combo_nose_length"),
            ("Genişlik", ["Normal", "Geniş", "Dar"], "combo_nose_width"),
            ("Kemer", ["Düz", "Kemerli", "Çökük"], "combo_nose_bridge"),
            ("Uç", ["Normal", "Kalkık", "Düşük"], "combo_nose_tip")
        ])
        
        # Lips
        create_group("Dudak Yapısı", [
            ("Üst Dudak", ["Normal", "Kalın", "İnce"], "combo_lips_upper"),
            ("Alt Dudak", ["Normal", "Kalın", "İnce"], "combo_lips_lower"),
            ("Genişlik", ["Normal", "Geniş", "Dar"], "combo_lips_width")
        ])
        
        # Chin
        create_group("Çene Yapısı", [
            ("Genişlik", ["Normal", "Geniş", "Dar", "Sivri"], "combo_chin_width"),
            ("Çıkıntı", ["Normal", "Çıkık", "Geride"], "combo_chin_prominence"),
            ("Gamze", ["Yok", "Var"], "combo_chin_dimple")
        ])
        
        # Ears
        create_group("Kulak Yapısı", [
            ("Boyut", ["Normal", "Büyük", "Küçük"], "combo_ears_size"),
            ("Açıklık", ["Normal", "Ayrık", "Yapışık"], "combo_ears_prominence"),
            ("Meme", ["Normal", "Ayrık", "Yapışık"], "combo_ears_lobe")
        ])
    
    def load_image_at_index(self, index):
        """Load image at specific index"""
        data = self.manager.get_image_at_index(index)
        if data:
            self.current_image_data = data
            self.current_index = index
            
            # Save current index to settings
            self.settings.setValue("last_index", index)
            
            # Update UI
            self.lbl_image_name.setText(data['image_name'])
            self.update_progress_label()
            
            # Load Image
            if isinstance(data['image'], np.ndarray):
                self.current_clean_image = data['image']
                h, w = self.current_clean_image.shape[:2]
                self.lbl_image_size.setText(f"{w}x{h}")
                
                # Auto-enable annotations on load
                self.btn_toggle_annotations.setChecked(True)
                self.btn_toggle_annotations.setText("👁️ Etiketleri Gizle")
                self.load_current_image() # This will handle drawing based on toggle state
                
                # Load existing annotations into combos
                if data['annotations']:
                    self.load_annotations_to_ui(data['annotations'])
                else:
                    # Reset combos or run auto-annotation if needed (User said remove auto button, maybe auto-run?)
                    # For now just reset
                    self.reset_ui()
            else:
                self.image_label.setText("Resim yüklenemedi")
        else:
            self.image_label.setText("Veri bulunamadı")
            
    def load_current_image(self):
        """Display current image with optional annotations"""
        if self.current_clean_image is None:
            return
            
        # Check toggle state
        show_annotations = self.btn_toggle_annotations.isChecked()
        
        display_img = self.current_clean_image.copy()
        
        if show_annotations:
            # Generate annotated image if not cached
            if self.current_annotated_image is None or not np.array_equal(self.current_annotated_image_source, self.current_clean_image):
                landmarks = self.annotator.get_landmarks(self.current_clean_image)
                if landmarks:
                    # Use Visualizer to draw
                    rgb_img = cv2.cvtColor(self.current_clean_image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    self.current_annotated_image = Visualizer.draw_measurements(pixmap, landmarks, self.annotator, self.current_clean_image)
                    self.current_annotated_image_source = self.current_clean_image # Cache the source image
                else:
                    # No landmarks found, just use clean image
                    self.current_annotated_image = QPixmap() # Empty or fallback
                    self.current_annotated_image_source = None
            
            if self.current_annotated_image and not self.current_annotated_image.isNull():
                 scaled = self.current_annotated_image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.image_label.setPixmap(scaled)
        else:
            # Display clean image
            self.display_image(self.current_clean_image)

    def display_image(self, img_array):
        """Display numpy image on label"""
        if img_array is None:
            return
            
        # Convert to QPixmap
        rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Ensure data is copied
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        
    def toggle_annotations(self):
        """Toggle annotation visibility"""
        if self.btn_toggle_annotations.isChecked():
            self.btn_toggle_annotations.setText("🙈 Etiketleri Gizle")
            self.btn_toggle_annotations.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 8px;")
            
            # Generate annotated image if not cached
            if self.current_annotated_image is None:
                landmarks = self.annotator.get_landmarks(self.current_clean_image)
                if landmarks:
                    # Use Visualizer to draw
                    rgb_img = cv2.cvtColor(self.current_clean_image, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    
                    self.current_annotated_image = Visualizer.draw_measurements(pixmap, landmarks, self.annotator, self.current_clean_image)
                else:
                    # No landmarks found, just use clean image
                    self.current_annotated_image = QPixmap() # Empty or fallback
            
            if self.current_annotated_image and not self.current_annotated_image.isNull():
                 scaled = self.current_annotated_image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 self.image_label.setPixmap(scaled)
        else:
            self.btn_toggle_annotations.setText("👁️ Etiketleri Göster")
            self.btn_toggle_annotations.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 8px;")
            self.display_image(self.current_clean_image)
            
    def resizeEvent(self, event):
        """Handle resize to keep image scaled"""
        super().resizeEvent(event)
        if self.current_clean_image is not None:
            # Re-display current state
            self.toggle_annotations() # Call without argument, it uses self.btn_toggle_annotations.isChecked()

    def load_annotations_to_ui(self, annotations):
        """Load annotations into UI controls"""
        # Face shape
        if 'face_shape' in annotations:
            self.set_combo_value(self.combo_face_shape, annotations['face_shape'].get('shape', 'Seçilmedi'))
        
        # Forehead
        if 'forehead' in annotations:
            self.set_combo_value(self.combo_forehead_width, annotations['forehead'].get('width', 'Seçilmedi'))
            self.set_combo_value(self.combo_forehead_height, annotations['forehead'].get('height', 'Seçilmedi'))
            self.set_combo_value(self.combo_forehead_slope, annotations['forehead'].get('slope', 'Seçilmedi'))
        
        # Eyes
        if 'eyes' in annotations:
            self.set_combo_value(self.combo_eyes_size, annotations['eyes'].get('size', 'Seçilmedi'))
            self.set_combo_value(self.combo_eyes_slant, annotations['eyes'].get('slant', 'Seçilmedi'))
            self.set_combo_value(self.combo_eyes_spacing, annotations['eyes'].get('spacing', 'Seçilmedi'))
            self.set_combo_value(self.combo_eyes_depth, annotations['eyes'].get('depth', 'Seçilmedi'))
        
        # Nose
        if 'nose' in annotations:
            self.set_combo_value(self.combo_nose_length, annotations['nose'].get('length', 'Seçilmedi'))
            self.set_combo_value(self.combo_nose_width, annotations['nose'].get('width', 'Seçilmedi'))
            self.set_combo_value(self.combo_nose_bridge, annotations['nose'].get('bridge', 'Seçilmedi'))
            self.set_combo_value(self.combo_nose_tip, annotations['nose'].get('tip', 'Seçilmedi'))
        
        # Lips
        if 'lips' in annotations:
            self.set_combo_value(self.combo_lips_upper, annotations['lips'].get('upper_thickness', 'Seçilmedi'))
            self.set_combo_value(self.combo_lips_lower, annotations['lips'].get('lower_thickness', 'Seçilmedi'))
            self.set_combo_value(self.combo_lips_width, annotations['lips'].get('width', 'Seçilmedi'))
        
        # Chin
        if 'chin' in annotations:
            self.set_combo_value(self.combo_chin_width, annotations['chin'].get('width', 'Seçilmedi'))
            self.set_combo_value(self.combo_chin_prominence, annotations['chin'].get('prominence', 'Seçilmedi'))
            self.set_combo_value(self.combo_chin_dimple, annotations['chin'].get('dimple', 'Seçilmedi'))
        
        # Ears
        if 'ears' in annotations:
            self.set_combo_value(self.combo_ears_size, annotations['ears'].get('size', 'Seçilmedi'))
            self.set_combo_value(self.combo_ears_prominence, annotations['ears'].get('prominence', 'Seçilmedi'))
            self.set_combo_value(self.combo_ears_lobe, annotations['ears'].get('lobe', 'Seçilmedi'))
    
    def set_combo_value(self, combo, value):
        """Set combobox to specific value"""
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
    
    def reset_ui(self):
        """Reset all UI controls to default"""
        for combo in [
            self.combo_face_shape, self.combo_forehead_width, self.combo_forehead_height, self.combo_forehead_slope,
            self.combo_eyes_size, self.combo_eyes_slant, self.combo_eyes_spacing, self.combo_eyes_depth,
            self.combo_nose_length, self.combo_nose_width, self.combo_nose_bridge, self.combo_nose_tip,
            self.combo_lips_upper, self.combo_lips_lower, self.combo_lips_width,
            self.combo_chin_width, self.combo_chin_prominence, self.combo_chin_dimple,
            self.combo_ears_size, self.combo_ears_prominence, self.combo_ears_lobe
        ]:
            combo.setCurrentIndex(0)  # 'Seçilmedi'
    
    def auto_annotate(self):
        """Generate automatic annotations using AutoAnnotator"""
        if self.current_clean_image is None:
            return
            
        try:
            # Delegate to AutoAnnotator
            annotations = self.annotator.annotate_image(self.current_clean_image)
            
            # Load to UI
            self.load_annotations_to_ui(annotations)
            print("Auto-annotation completed successfully.")
            
        except Exception as e:
            print(f"Error in auto_annotate: {e}")
            import traceback
            traceback.print_exc()

    def batch_auto_annotate(self):
        """Automatically annotate all images from current index to end"""
        total = self.manager.get_total_count()
        remaining = total - self.current_index
        
        if remaining <= 0:
            QMessageBox.information(self, "Bilgi", "İşlenecek fotoğraf kalmadı.")
            return

        # Confirmation
        reply = QMessageBox.question(
            self, 
            "Toplu Etiketleme Onayı", 
            f"Şu anki fotoğraftan başlayarak kalan {remaining} fotoğraf otomatik olarak etiketlenecek ve kaydedilecek.\n\n"
            "Mevcut manuel etiketler üzerine yazılabilir.\n\nDevam etmek istiyor musunuz?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        # Progress Dialog
        progress = QProgressDialog("Fotoğraflar işleniyor...", "İptal", 0, remaining, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        processed_count = 0
        
        try:
            # Start from current index
            start_idx = self.current_index
            
            for i in range(start_idx, total):
                if progress.wasCanceled():
                    break
                
                # Update progress text
                progress.setLabelText(f"İşleniyor: {i+1}/{total}")
                
                # Get image data directly from manager (avoid loading into UI for speed)
                data = self.manager.get_image_at_index(i)
                if not data or not isinstance(data['image'], np.ndarray):
                    continue
                
                image = data['image']
                image_name = data['image_name']
                
                # Run Auto Annotator
                annotations = self.annotator.annotate_image(image)
                
                # Save to DB
                self.manager.save_annotation(image_name, annotations)
                
                processed_count += 1
                progress.setValue(processed_count)
                QApplication.processEvents()
                
            QMessageBox.information(self, "Tamamlandı", f"{processed_count} fotoğraf başarıyla etiketlendi ve kaydedildi.")
            
            # Reload current image to show updates
            self.load_image_at_index(self.current_index)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"İşlem sırasında hata oluştu:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()
    
    def save_annotation(self):
        """Save current annotations"""
        if self.current_image_data is None:
            return
        
        # Collect annotations from UI
        annotations = {
            'face_shape': {'shape': self.combo_face_shape.currentText()},
            'forehead': {
                'width': self.combo_forehead_width.currentText(),
                'height': self.combo_forehead_height.currentText(),
                'slope': self.combo_forehead_slope.currentText()
            },
            'eyes': {
                'size': self.combo_eyes_size.currentText(),
                'slant': self.combo_eyes_slant.currentText(),
                'spacing': self.combo_eyes_spacing.currentText(),
                'depth': self.combo_eyes_depth.currentText()
            },
            'nose': {
                'length': self.combo_nose_length.currentText(),
                'width': self.combo_nose_width.currentText(),
                'bridge': self.combo_nose_bridge.currentText(),
                'tip': self.combo_nose_tip.currentText()
            },
            'lips': {
                'upper_thickness': self.combo_lips_upper.currentText(),
                'lower_thickness': self.combo_lips_lower.currentText(),
                'width': self.combo_lips_width.currentText()
            },
            'chin': {
                'width': self.combo_chin_width.currentText(),
                'prominence': self.combo_chin_prominence.currentText(),
                'dimple': self.combo_chin_dimple.currentText()
            },
            'ears': {
                'size': self.combo_ears_size.currentText(),
                'prominence': self.combo_ears_prominence.currentText(),
                'lobe': self.combo_ears_lobe.currentText()
            },
            'auto_generated': False
        }
        
        # Save to DB
        image_name = self.current_image_data['image_name']
        self.manager.save_annotation(image_name, annotations)
        
        # Update status label (Feedback in status bar area)
        status = self.manager.get_status()
        self.label_progress.setText(f"✅ Kaydedildi! ({status['annotated']}/{status['total']})")
        self.label_progress.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6e3a1;")
        
        # Revert status text after delay
        QTimer.singleShot(2000, lambda: self.update_progress_label())
        
        # Button feedback (subtle)
        # self.btn_save.setStyleSheet("background-color: #94e2d5; color: #1e1e2e; padding: 12px; font-weight: bold; font-size: 14px;")
        # QTimer.singleShot(500, lambda: self.btn_save.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; padding: 12px; font-weight: bold; font-size: 14px;"))
        
    def update_progress_label(self):
        """Update progress label text"""
        total = self.manager.get_total_count()
        self.label_progress.setText(f"{self.current_index + 1} / {total}")
        self.label_progress.setStyleSheet("font-size: 14px; font-weight: bold; color: #cdd6f4;")

    def flash_button(self, button):
        """Visual feedback for button click"""
        original_style = button.styleSheet()
        # Brighten background
        button.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; padding: 10px; font-weight: bold;")
        QTimer.singleShot(200, lambda: button.setStyleSheet(original_style))

    def prev_image(self):
        """Go to previous image"""
        self.flash_button(self.btn_prev)
        self.save_annotation() # Save before moving
        if self.current_index > 0:
            self.load_image_at_index(self.current_index - 1)
    
    def next_image(self):
        """Go to next image"""
        self.flash_button(self.btn_next)
        self.save_annotation() # Save before moving
        if self.current_index < self.manager.get_total_count() - 1:
            self.load_image_at_index(self.current_index + 1)
            
    def run_migration(self):
        """Run manual migration"""
        self.btn_migrate.setEnabled(False)
        self.btn_migrate.setText("⏳ Aktarılıyor...")
        
        # Run in background ideally, but for now blocking is safer to avoid race conditions
        count = self.manager.migrate_from_files()
        
        self.btn_migrate.setText(f"✅ {count} Aktarıldı")
        self.btn_migrate.setEnabled(True)
        
        # Refresh current view
        self.load_image_at_index(self.current_index)
        QTimer.singleShot(2000, lambda: self.btn_migrate.setText("📂 Verileri İçe Aktar"))

    def export_dataset_ui(self):
        """Trigger dataset export from UI"""
        try:
            from scripts.export_dataset import export_dataset
            
            # Show confirmation
            reply = QMessageBox.question(
                self, 'Veri Setini Dışa Aktar',
                "Veritabanındaki etiketli veriler 'dataset/export' klasörüne aktarılacak.\n\nDevam etmek istiyor musunuz?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Show progress dialog (indefinite)
                progress = QProgressDialog("Veriler dışa aktarılıyor...", None, 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                QApplication.processEvents()
                
                # Run export
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(base_dir, "dataset", "export")
                
                export_dataset(output_dir)
                
                progress.close()
                
                QMessageBox.information(
                    self, "Başarılı", 
                    f"Veri seti başarıyla dışa aktarıldı!\n\nKonum: {output_dir}\n\nArtık AI Eğitimi ekranından model eğitebilirsiniz."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dışa aktarma sırasında hata oluştu:\n{str(e)}")





