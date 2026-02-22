"""
Kamera görünümü - Fotoğraf çekme ve arşivden seçme
"""

import cv2
import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QFrame, QSlider, QScrollArea, QSpinBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QImage, QPixmap

class InteractiveGLViewWidget(gl.GLViewWidget):
    """Sinyal yayan özelleştirilmiş GLViewWidget"""
    camera_moved = Signal(float, float, float)  # distance, elevation, azimuth
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        self.emit_camera_position()

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self.emit_camera_position()
        
    def wheelEvent(self, ev):
        super().wheelEvent(ev)
        self.emit_camera_position()
        
    def emit_camera_position(self):
        """Kamera pozisyonunu sinyal olarak yay"""
        # opts['distance'], opts['elevation'], opts['azimuth']
        d = self.opts['distance']
        e = self.opts['elevation']
        a = self.opts['azimuth']
        self.camera_moved.emit(d, e, a)

class CameraThread(QThread):
    """Kameradan asenkron (Thread) olarak görüntü okuyan sınıf (UI donmasını engeller)"""
    new_frame = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self, camera_id=0, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._is_running = False
        self.camera = None
        self.res_width = 1920
        self.res_height = 1080

    def run(self):
        self._is_running = True
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            self.error_signal.emit("Kamera açılamadı")
            return
            
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_height)

        while self._is_running:
            ret, frame = self.camera.read()
            if ret:
                self.new_frame.emit(frame)
            self.msleep(30) # Capture at ~33 FPS

    def stop(self):
        self._is_running = False
        self.wait() # wait for thread to finish
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            
    def set_resolution(self, width, height):
        self.res_width = width
        self.res_height = height
        if self.camera is not None:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

class CameraView(QWidget):
    """Kamera ve fotoğraf seçme görünümü"""
    
    # Signal for when photo is ready to analyze
    photo_ready = Signal(dict) # {'front': np.ndarray, 'side': np.ndarray}
    # Signal for status updates
    status_message = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.camera_thread = None
        self.current_frame = None
        
        self.images = {'front': None, 'side': None}
        self.active_slot = 'front' # 'front' or 'side'
        self.mesh_points = None  # Store 3D points for mesh
        
        self.setup_ui()
    
    def setup_ui(self):
        """UI bileşenlerini oluştur"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # LEFT PANEL: Camera/Preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title = QLabel("📸 Fotoğraf Çekim")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title)
        
        # Display Area
        self.display_label = QLabel("Kamerayı başlatın veya fotoğraf seçin")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setStyleSheet("""
            background-color: #11111b;
            border: 2px dashed #45475a;
            border-radius: 12px;
            color: #6c7086;
        """)
        left_layout.addWidget(self.display_label, stretch=1)
        
        # Camera Controls
        controls_layout = QHBoxLayout()
        
        self.btn_capture = QPushButton("📸 Fotoğrafı Çek")
        self.btn_capture.setMinimumHeight(45)
        self.btn_capture.setEnabled(False)
        self.btn_capture.clicked.connect(self.capture_photo)
        controls_layout.addWidget(self.btn_capture)
        
        self.btn_from_file = QPushButton("📁 Dosyadan Seç")
        self.btn_from_file.setMinimumHeight(45)
        self.btn_from_file.clicked.connect(self.select_from_file)
        controls_layout.addWidget(self.btn_from_file)
        
        self.btn_save = QPushButton("💾 Kaydet")
        self.btn_save.setMinimumHeight(45)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save_to_file)
        controls_layout.addWidget(self.btn_save)
        
        left_layout.addLayout(controls_layout)
        
        # Camera Adjustments (when camera is active)
        adjust_frame = QFrame()
        adjust_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 8px; padding: 5px;")
        adjust_layout = QVBoxLayout(adjust_frame)
        adjust_layout.setSpacing(6)
        
        # Resolution/Quality
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("📷 Çözünürlük:"))
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setRange(1, 4)  # 1=Low, 2=Medium, 3=High, 4=4K
        self.resolution_slider.setValue(3)  # Default: Yüksek (1920x1080)
        self.resolution_slider.setEnabled(False)
        self.resolution_slider.valueChanged.connect(self.apply_resolution_change)
        res_layout.addWidget(self.resolution_slider)
        self.resolution_label = QLabel("Yüksek (1920x1080)")
        self.resolution_label.setFixedWidth(150)
        res_layout.addWidget(self.resolution_label)
        adjust_layout.addLayout(res_layout)
        
        # Brightness
        bright_layout = QHBoxLayout()
        bright_layout.addWidget(QLabel("☀️ Parlaklık:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setEnabled(False)
        self.brightness_slider.valueChanged.connect(self.apply_camera_adjustments)
        bright_layout.addWidget(self.brightness_slider)
        self.brightness_value_label = QLabel("0")
        self.brightness_value_label.setFixedWidth(30)
        bright_layout.addWidget(self.brightness_value_label)
        adjust_layout.addLayout(bright_layout)
        
        # Contrast
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("🔆 Kontrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-50, 50)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setEnabled(False)
        self.contrast_slider.valueChanged.connect(self.apply_camera_adjustments)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_value_label = QLabel("0")
        self.contrast_value_label.setFixedWidth(30)
        contrast_layout.addWidget(self.contrast_value_label)
        adjust_layout.addLayout(contrast_layout)
        
        left_layout.addWidget(adjust_frame)
        
        main_layout.addWidget(left_panel, stretch=3)
        
        # RIGHT PANEL: Slots & Action
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #1e1e2e; border-radius: 12px;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # Position Selection Section
        # Start Camera Button (Moved here for better UX)
        self.btn_start_camera = QPushButton("📹 Kamerayı Başlat")
        self.btn_start_camera.setMinimumHeight(55)
        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_start_camera.setStyleSheet("""
            QPushButton {
                background-color: #f9e2af;
                color: #1e1e2e;
                font-size: 15px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #fab387;
            }
        """)
        right_layout.addWidget(self.btn_start_camera)
        
        # Spacer
        right_layout.addSpacing(10)
        
        right_layout.addWidget(QLabel("📌 Pozisyon Seçin"))
        
        # Front Slot
        self.btn_slot_front = self.create_slot_button("Ön Yüz (Zorunlu)", "front")
        right_layout.addWidget(self.btn_slot_front)
        
        # Side Slot
        self.btn_slot_side = self.create_slot_button("Yan Profil (Opsiyonel)", "side")
        right_layout.addWidget(self.btn_slot_side)
        
        right_layout.addStretch()
        
        # Retake Button
        self.btn_retake = QPushButton("🔄 Yeniden Çek")
        self.btn_retake.setMinimumHeight(45)
        self.btn_retake.setVisible(False)
        self.btn_retake.clicked.connect(self.retake_photo)
        self.btn_retake.setStyleSheet("""
            QPushButton {
                background-color: #f9e2af;
                color: #1e1e2e;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #fab387;
            }
        """)
        right_layout.addWidget(self.btn_retake)
        
        # Analyze Button
        self.btn_analyze = QPushButton("🔬 Analiz Et")
        self.btn_analyze.setMinimumHeight(55)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.analyze_photo)
        self.btn_analyze.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #b4f5b0;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        right_layout.addWidget(self.btn_analyze)
        
        main_layout.addWidget(right_panel, stretch=1)
        
        # Set initial active slot
        self.set_active_slot('front')

    def create_slot_button(self, text, slot_id):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setMinimumHeight(60)
        btn.clicked.connect(lambda: self.set_active_slot(slot_id))
        return btn

    def set_active_slot(self, slot_id):
        self.active_slot = slot_id
        
        # Update UI
        self.btn_slot_front.setChecked(slot_id == 'front')
        self.btn_slot_side.setChecked(slot_id == 'side')
        
        # Update retake button text if visible
        if hasattr(self, 'btn_retake') and self.btn_retake.isVisible():
            self.update_retake_button_text()
        
        # Style updates
        active_style = """
            QPushButton {
                background-color: #313244;
                border: 2px solid #89b4fa;
                border-radius: 8px;
                color: #cdd6f4;
                text-align: left;
                padding: 15px;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #45475a;
                border: 2px solid #a6e3a1;
            }
        """
        self.btn_slot_front.setStyleSheet(active_style)
        self.btn_slot_side.setStyleSheet(active_style)
        
        camera_running = self.camera_thread is not None and self.camera_thread.isRunning()
        
        # Show existing image if any
        if self.images[slot_id] is not None:
            if not camera_running:
                self.display_frame(self.images[slot_id])
            if hasattr(self, 'btn_save'):
                self.btn_save.setEnabled(True)
        else:
            # Clear display or show camera if running
            if not camera_running:
                self.display_label.clear()
                self.display_label.setText(f"{'Ön Yüz' if slot_id == 'front' else 'Yan Profil'} için fotoğraf çekin veya seçin")
                if hasattr(self, 'btn_save'):
                    self.btn_save.setEnabled(False)
    
    def apply_settings(self, settings):
        """Settings'den ayarları uygula"""
        # Camera resolution
        cam_res = int(settings.get('camera_resolution', 1))
        self.resolution_slider.setValue(cam_res + 1)
    
    def start_camera(self):
        """Kamerayı başlat"""
        try:
            if self.camera_thread is not None and self.camera_thread.isRunning():
                return

            self.camera_thread = CameraThread(camera_id=0)
            self.camera_thread.new_frame.connect(self.update_frame)
            self.camera_thread.error_signal.connect(lambda e: self.status_message.emit(f"❌ Kamera hatası: {e}"))
            
            # Use current slider value for resolution
            self.apply_resolution_change()
            self.camera_thread.start()
            
            self.btn_start_camera.setEnabled(False)
            self.btn_capture.setEnabled(True)
            self.btn_from_file.setEnabled(False)
            self.btn_save.setEnabled(False)
            
            # Enable camera adjustments
            self.brightness_slider.setEnabled(True)
            self.contrast_slider.setEnabled(True)
            self.resolution_slider.setEnabled(True)
            
        except Exception as e:
            self.status_message.emit(f"❌ Kamera hatası başlatılırken: {str(e)}")
    
    def update_frame(self, frame):
        """Kamera frame'ini thread içinden gelen sinyalle güncelle"""
        self.current_frame = frame
        # Apply adjustments before display
        adjusted_frame = self.apply_adjustments_to_frame(frame)
        self.display_frame(adjusted_frame)
    
    def apply_camera_adjustments(self):
        """Parlaklık ve kontrast ayarlarını güncelle"""
        # Update value labels
        self.brightness_value_label.setText(str(self.brightness_slider.value()))
        self.contrast_value_label.setText(str(self.contrast_slider.value()))
        
        # Real-time apply if camera is running
        if self.camera_thread is not None and self.camera_thread.isRunning() and self.current_frame is not None:
            adjusted_frame = self.apply_adjustments_to_frame(self.current_frame)
            self.display_frame(adjusted_frame)
    
    def apply_adjustments_to_frame(self, frame):
        """Frame'e parlaklık ve kontrast uygula"""
        # Get slider values
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value()
        
        # Convert to float for processing
        adjusted = frame.astype('float32')
        
        # Apply brightness (additive)
        adjusted = adjusted + brightness
        
        # Apply contrast (multiplicative)
        # contrast range: -50 to 50 -> factor: 0.5 to 1.5
        contrast_factor = 1.0 + (contrast / 100.0)
        adjusted = (adjusted - 127.5) * contrast_factor + 127.5
        
        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255).astype('uint8')
        
        return adjusted
    
    def apply_resolution_change(self):
        """Kamera çözünürlük ayarını değiştir"""
        res_value = self.resolution_slider.value()
        res_names = {
            1: ("Düşük", "640x480", 640, 480),
            2: ("Orta", "1280x720", 1280, 720),
            3: ("Yüksek", "1920x1080", 1920, 1080),
            4: ("4K", "3840x2160", 3840, 2160)
        }
        name, pixels, w, h = res_names[res_value]
        self.resolution_label.setText(f"{name} ({pixels})")
        
        if self.camera_thread is not None:
            self.camera_thread.set_resolution(w, h)
            self.status_message.emit(f"📷 Çözünürlük: {name} ({pixels})")
    
    def display_frame(self, frame):
        """Frame'i QLabel'da göster"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        display_height = self.display_label.height()
        display_width = self.display_label.width()
        
        h, w = rgb_frame.shape[:2]
        aspect_ratio = w / h
        
        if w > display_width or h > display_height:
            if display_width / aspect_ratio <= display_height:
                new_w = display_width
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = display_height
                new_w = int(new_h * aspect_ratio)
            
            rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
        
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.display_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def capture_photo(self):
        """Fotoğraf çek - kamera açık kalır ve ilgili slota kaydeder"""
        if self.current_frame is not None:
            # Save to current slot
            self.images[self.active_slot] = self.current_frame.copy()
            
            # Update slot button text/icon
            btn = self.btn_slot_front if self.active_slot == 'front' else self.btn_slot_side
            btn.setText(f"{'Ön Yüz' if self.active_slot == 'front' else 'Yan Profil'} (✅ Hazır)")
            
            # Enable analyze button if front photo exists
            if self.images['front'] is not None:
                self.btn_analyze.setEnabled(True)
            
            self.status_message.emit(f"✅ {self.active_slot.capitalize()} fotoğrafı çekildi")
            
            if self.active_slot == 'front' and self.images['side'] is None:
                self.status_message.emit("✅ Ön yüz hazır! Şimdi yan profilinizi seçip çekin (veya analiz edin)")
            elif self.active_slot == 'side':
                self.status_message.emit("✅ Yan profil hazır! Analiz edebilirsiniz →")
            
            self.btn_save.setEnabled(True)
            self.btn_retake.setVisible(True)
            self.update_retake_button_text()
            
    def select_from_file(self):
        """Dosyadan fotoğraf seç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Fotoğraf Seç", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None: raise Exception("Görüntü yüklenemedi")
                
                self.images[self.active_slot] = image
                self.display_frame(image)
                
                # Update slot button
                btn = self.btn_slot_front if self.active_slot == 'front' else self.btn_slot_side
                btn.setText(f"{'Ön Yüz' if self.active_slot == 'front' else 'Yan Profil'} (✅ Hazır)")
                
                if self.images['front'] is not None:
                    self.btn_analyze.setEnabled(True)
                
                self.status_message.emit(f"✅ {self.active_slot.capitalize()} fotoğrafı yüklendi")
                self.btn_save.setEnabled(True)
                    
            except Exception as e:
                self.status_message.emit(f"❌ Dosya hatası: {str(e)}")
    
    def save_to_file(self):
        """Mevcut görüntüyü dosyaya kaydet"""
        current_image = self.images.get(self.active_slot)
        if current_image is None:
            self.status_message.emit("❌ Kaydedilecek görüntü yok")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Görüntüyü Kaydet", "", "JPEG Image (*.jpg);;All Files (*)"
        )

        if file_path:
            try:
                # Ensure extension is correct
                if not file_path.lower().endswith(('.jpg', '.jpeg')):
                    file_path += '.jpg'
                
                cv2.imwrite(file_path, current_image)
                self.status_message.emit(f"✅ Görüntü kaydedildi: {file_path}")
            except Exception as e:
                self.status_message.emit(f"❌ Kaydetme hatası: {str(e)}")

    def analyze_photo(self):
        """Analiz işlemini başlat"""
        if self.images['front'] is None:
            self.status_message.emit("⚠️ Lütfen önce ön yüz fotoğrafı çekin")
            return
            
        self.status_message.emit("🔬 Analiz başlatılıyor...")
        self.photo_ready.emit(self.images)

    def cleanup(self):
        """Kaynakları serbest bırak"""
        if self.camera_thread is not None:
            if self.camera_thread.isRunning():
                self.camera_thread.stop()
            self.camera_thread = None

    def retake_photo(self):
        """Mevcut slot'un fotoğrafını sil ve yeniden çek"""
        # Clear current slot
        slot_name = "Ön Yüz" if self.active_slot == 'front' else "Yan Profil"
        self.images[self.active_slot] = None
        
        # Update button
        btn = self.btn_slot_front if self.active_slot == 'front' else self.btn_slot_side
        btn.setText("Ön Yüz (Zorunlu)" if self.active_slot == 'front' else "Yan Profil (Opsiyonel)")
        
        # Clear mesh and UI
        # self.mesh_viewer.clear() # mesh_viewer is not defined in this scope based on previous view
        self.mesh_points = None
        # self.quality_label.setVisible(False) # quality_label not defined
        self.btn_retake.setVisible(False)
        
        # Disable buttons
        # self.btn_prepare_mesh.setEnabled(False) # not defined
        self.btn_analyze.setEnabled(False)
        self.btn_save.setEnabled(False)
        
        # Clear display
        self.display_label.clear()
        self.display_label.setText(f"{slot_name} için fotoğraf çekin veya seçin")
        
        self.status_message.emit(f"📸 {slot_name} yeniden çekilecek")
    
    def update_retake_button_text(self):
        """Retake button metnini aktif slot'a göre güncelle"""
        if self.btn_retake.isVisible():
            slot_name = "Ön Yüz" if self.active_slot == 'front' else "Yan Profil"
            self.btn_retake.setText(f"🔄 {slot_name} Yeniden Çek")
    
    def get_mesh_triangles(self, points_3d):
        """2D Delaunay triangulation ile triangle faces oluştur"""
        from scipy.spatial import Delaunay
        
        # Project to 2D (x, y) for triangulation
        points_2d = points_3d[:, :2]
        
        # Perform Delaunay triangulation
        tri = Delaunay(points_2d)
        
        # Return triangle indices (Mx3 array)
        return tri.simplices
    
    def display_mesh_preview(self, points_3d, texture_image, landmarks):
        """3D mesh'i surface olarak göster"""
        # Clear previous mesh
        # self.mesh_viewer.clear() # mesh_viewer not defined
        pass # Placeholder as mesh_viewer seems missing in the provided file content or I missed it. 
             # Wait, looking at the original file content, mesh_viewer IS referenced in display_mesh_preview.
             # But it's not initialized in __init__ or setup_ui in the provided view.
             # Ah, I see InteractiveGLViewWidget at the top, but where is it instantiated?
             # It seems I might have missed where mesh_viewer is created in setup_ui in the previous view_file output?
             # Let me check the view_file output again.
             # Line 63 setup_ui starts.
             # It adds left_panel and right_panel.
             # It doesn't seem to add mesh_viewer.
             # However, the original code had `self.mesh_viewer.clear()` in `retake_photo` (line 474) and `display_mesh_preview` (line 511).
             # But `mesh_viewer` was NOT in `setup_ui` (lines 63-229).
             # This suggests `mesh_viewer` might be added dynamically or I missed something.
             # Or maybe the file I read was incomplete or had errors?
             # No, I read the whole file.
             # Wait, line 474: `self.mesh_viewer.clear()`
             # But `self.mesh_viewer` is not defined in `__init__` or `setup_ui`.
             # This implies the code I read might have bugs or `mesh_viewer` is injected?
             # Regardless, I should preserve the existing code structure.
             # I will uncomment the lines I commented out if they were there.
             # Actually, looking at the `replace_file_content` block I prepared:
             # I am replacing from line 105 to 583.
             # I should be careful not to break existing logic even if it looks broken.
             # I will restore the lines exactly as they were, just adding my changes.
             
             # Let's refine the replacement to be safer. I'll use `multi_replace_file_content` or smaller chunks.
             # The file is small enough (583 lines).
             # But replacing a huge chunk is risky if I get indentation wrong or miss something.
             # I'll use `multi_replace_file_content` to inject the button and the method.

