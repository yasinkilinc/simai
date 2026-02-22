"""
Detaylı analiz görünümü
3D mesh, heatmap, özellik kartları
"""

import cv2
import logging
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QSplitter,
    QTabWidget, QGridLayout, QSizePolicy, QProgressBar,
    QGroupBox, QFormLayout, QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PIL import Image, ImageDraw, ImageFont
import platform
from desktop_app.feedback_dialog import FeedbackDialog
import sys
import os
# Add parent directory to path for imports if not already there
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization import Visualizer
from annotation_engine import AutoAnnotator


class AnalysisView(QWidget):
    """Detaylı analiz sonuçları görünümü"""
    
    # Signal for status updates
    status_message = Signal(str)
    analysis_saved = Signal() # Signal emitted when analysis is saved

    def __init__(self):
        super().__init__()
        self.current_report = None
        self.current_image = None
        self.side_image = None
        self.points_3d = None
        self.landmarks = None
        self.current_heatmap_image = None
        self.current_loaded_id = None # ID of the currently loaded analysis (for updates)
        
        self.setup_ui()
    
    def setup_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("📊 Analiz Sonuçları")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Content Container
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        
        # 1. Visuals Section (Top) - Fixed height or ratio
        visuals_container = QWidget()
        visuals_layout = QHBoxLayout(visuals_container)
        visuals_layout.setContentsMargins(0, 0, 0, 0)
        visuals_layout.setSpacing(10)
        
        self.create_visuals_section(visuals_layout)
        content_layout.addWidget(visuals_container, stretch=4)
        
        # 2. Data Section (Bottom)
        data_container = QWidget()
        data_layout = QVBoxLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(5)
        
        self.create_data_section(data_layout)
        content_layout.addWidget(data_container, stretch=6)
        
        layout.addWidget(content_widget)
        
        # Create history panel (collapsible at bottom) - REMOVED as requested
        # history_panel = self.create_history_panel()
        # layout.addWidget(history_panel)
        
    def showEvent(self, event):
        """Widget gösterildiğinde çağrılır - history'i refresh et"""
        super().showEvent(event)
        if hasattr(self, 'refresh_history') and hasattr(self, 'history_layout'):
            # self.refresh_history() # Removed history panel
            pass
    
    def create_visuals_section(self, parent_layout):
        """Görsel bölümlerini (heatmap ve 3D) oluştur"""
        # Heatmap Section
        self.create_heatmap_section(parent_layout)
        
        # 3D Section
        self.create_3d_section(parent_layout)

    def create_heatmap_section(self, parent_layout):
        """Isı haritası bölümünü oluştur"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Tab Widget for Images
        img_tabs = QTabWidget()
        img_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                background: #1e1e2e;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 4px 8px;
                min-width: 60px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)

        # Controls Header
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 5)
        
        # Toggle button for annotations (more reliable than checkbox)
        self.btn_toggle_annotations = QPushButton("👁️ Etiketleri Göster")
        self.btn_toggle_annotations.setCheckable(True)
        self.btn_toggle_annotations.setChecked(False)
        self.btn_toggle_annotations.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 2px solid #45475a;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:checked {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-color: #89b4fa;
            }
        """)
        self.btn_toggle_annotations.clicked.connect(self.toggle_annotations)
        controls_layout.addWidget(self.btn_toggle_annotations)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)

        # Front Tab
        front_tab = QWidget()
        front_layout = QVBoxLayout(front_tab)
        front_layout.setContentsMargins(0, 0, 0, 0)
        
        self.img_label = QLabel("Analiz bekleniyor...")
        self.img_label.setAlignment(Qt.AlignCenter)
        # self.img_label.setScaledContents(True) # Removed to keep aspect ratio
        self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.img_label.setStyleSheet("background-color: #11111b; border-radius: 4px;")
        front_layout.addWidget(self.img_label)
        
        img_tabs.addTab(front_tab, "Ön Profil")
        
        # Side Tab
        side_tab = QWidget()
        side_layout = QVBoxLayout(side_tab)
        side_layout.setContentsMargins(0, 0, 0, 0)
        
        self.side_img_label = QLabel("Yan profil yok")
        self.side_img_label.setAlignment(Qt.AlignCenter)
        # self.side_img_label.setScaledContents(True) # Removed to keep aspect ratio
        self.side_img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.side_img_label.setStyleSheet("background-color: #11111b; border-radius: 4px;")
        side_layout.addWidget(self.side_img_label)
        
        img_tabs.addTab(side_tab, "Yan Profil")
        
        layout.addWidget(img_tabs)
        
        parent_layout.addWidget(container, stretch=1)

    def create_3d_section(self, parent_layout):
        """3D model bölümünü oluştur"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label = QLabel("🎭 3D Yüz Modeli")
        label.setStyleSheet("font-weight: bold; color: #cdd6f4;")
        layout.addWidget(label)
        
        # 3D widget
        self.view_3d_container = QWidget()
        v3d_layout = QVBoxLayout(self.view_3d_container)
        v3d_layout.setContentsMargins(0,0,0,0)
        
        self.lbl_3d_warning = QLabel("3D Model ve derinlik analizi için\nlütfen yan profil fotoğrafı da yükleyin.")
        self.lbl_3d_warning.setAlignment(Qt.AlignCenter)
        self.lbl_3d_warning.setStyleSheet("color: #f9e2af; font-size: 13px; background-color: #11111b; border-radius: 6px; padding: 20px;")
        self.lbl_3d_warning.hide()
        v3d_layout.addWidget(self.lbl_3d_warning)

        self.view_3d = gl.GLViewWidget()
        self.view_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_3d.setCameraPosition(distance=500)
        self.view_3d.setBackgroundColor('#11111b')
        v3d_layout.addWidget(self.view_3d)
        
        layout.addWidget(self.view_3d_container)
        
        # 3D controls (Compact)
        self.controls_3d_widget = QWidget()
        controls_layout = QHBoxLayout(self.controls_3d_widget)
        controls_layout.setContentsMargins(0,0,0,0)
        controls_layout.setSpacing(5)
        
        self.btn_rotate = QPushButton("🔄 Döndür")
        self.btn_rotate.setCheckable(True)
        self.btn_rotate.setFixedHeight(25)
        self.btn_rotate.setStyleSheet("font-size: 11px; padding: 2px 8px; background-color: #313244; color: #cdd6f4; border-radius: 4px;")
        self.btn_rotate.clicked.connect(self.toggle_rotation)
        controls_layout.addWidget(self.btn_rotate)
        
        self.btn_reset = QPushButton("↺ Sıfırla")
        self.btn_reset.setFixedHeight(25)
        self.btn_reset.setStyleSheet("font-size: 11px; padding: 2px 8px; background-color: #313244; color: #cdd6f4; border-radius: 4px;")
        self.btn_reset.clicked.connect(self.reset_3d_view)
        controls_layout.addWidget(self.btn_reset)
        
        controls_layout.addStretch()
        layout.addWidget(self.controls_3d_widget)
        
        # Rotation timer
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.rotate_3d)
        self.rotation_angle = 0
        
        parent_layout.addWidget(container, stretch=1)
    
    def create_data_section(self, parent_layout):
        """Veri ve sekmeler bölümünü oluştur"""
        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                background: #1e1e2e;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 6px 12px;
                min-width: 80px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-size: 12px;
            }
            QTabBar::tab:selected {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
        """)
        
        # TAB 1: Özet (Compact)
        self.summary_tab = QWidget()
        self.create_summary_tab()
        self.tabs.addTab(self.summary_tab, "📊 Özet")
        
        # TAB 2: Detaylar (Compact)
        self.details_tab = QWidget()
        self.create_details_tab()
        self.tabs.addTab(self.details_tab, "📝 Detaylar")
        
        parent_layout.addWidget(self.tabs)
        
        # Action Buttons
        btn_container = QHBoxLayout()
        btn_container.addStretch()
        
        self.btn_save = QPushButton("💾 Kaydet")
        self.btn_save.setFixedSize(150, 35) # Compact size
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #b4f5b0;
            }
        """)
        self.btn_save.clicked.connect(self.save_current_analysis)
        btn_container.addWidget(self.btn_save)
        
        self.btn_feedback = QPushButton("✏️ Düzelt (AI)")
        self.btn_feedback.setFixedSize(150, 35)
        self.btn_feedback.setStyleSheet("""
            QPushButton {
                background-color: #f9e2af;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #fbe5b8;
            }
        """)
        self.btn_feedback.clicked.connect(self.open_feedback)
        btn_container.addWidget(self.btn_feedback)

        self.btn_export = QPushButton("📤 Dışa Aktar")
        self.btn_export.setFixedSize(150, 35)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #98c3ff;
            }
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
        """)
        btn_container.addWidget(self.btn_export)

        btn_container.addStretch()
        
        parent_layout.addLayout(btn_container)

    def save_current_analysis(self):
        """Mevcut analizi veritabanına BLOB olarak kaydet"""
        print("DEBUG: save_current_analysis called")
        if not self.current_report:
            print("DEBUG: No current report")
            return
        if self.clean_image is None:
            print("DEBUG: No clean image")
            return
            
        try:
            from desktop_app.database import Database
            import logging
            import numpy as np
            import cv2
            
            db = Database()
            
            # Prepare images (convert QPixmap to numpy if needed)
            def to_numpy(img):
                if img is None:
                    return None
                if isinstance(img, np.ndarray):
                    return img
                elif isinstance(img, QPixmap):
                    # Convert QPixmap to numpy array properly handling stride
                    qimg = img.toImage()
                    qimg = qimg.convertToFormat(QImage.Format_RGB888)
                    width = qimg.width()
                    height = qimg.height()
                    bytes_per_line = qimg.bytesPerLine()
                    ptr = qimg.bits()
                    
                    # Create array with proper stride
                    arr = np.array(ptr).reshape(height, bytes_per_line)
                    # Extract only the image data (remove padding if any)
                    arr = arr[:, :width * 3].reshape(height, width, 3)
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return None
            
            front_img = to_numpy(self.clean_image)
            side_img = to_numpy(self.side_image) if hasattr(self, 'side_image') else None
            heatmap_img = to_numpy(self.current_heatmap_image) if hasattr(self, 'current_heatmap_image') else None
            
            if front_img is None:
                logging.error("Failed to convert front image to numpy")
                self.status_message.emit("❌ Kayıt hatası: Ön görüntü dönüştürülemedi.")
                return
            
            # Prepare landmarks
            landmarks_serializable = []
            if hasattr(self, 'landmarks') and self.landmarks is not None:
                if hasattr(self.landmarks, 'landmark'):
                    for lm in self.landmarks.landmark:
                        landmarks_serializable.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                elif isinstance(self.landmarks, list):
                    if len(self.landmarks) > 0:
                        if isinstance(self.landmarks[0], dict):
                            landmarks_serializable = self.landmarks
                elif isinstance(self.landmarks, np.ndarray):
                    landmarks_serializable = self.landmarks.tolist()
                else:
                    # Handle Protobuf RepeatedCompositeFieldContainer (MediaPipe)
                    # It behaves like a list but isn't an instance of list
                    try:
                        # Try to iterate and check for x, y attributes
                        temp_list = []
                        for lm in self.landmarks:
                            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                                temp_list.append({'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0)})
                        
                        if temp_list:
                            landmarks_serializable = temp_list
                    except:
                        pass
            
            # Check if update or new save
            is_update = hasattr(self, 'current_loaded_id') and self.current_loaded_id
            
            if is_update:
                # Update existing record
                db.update_analysis(
                    self.current_loaded_id,
                    front_img,
                    self.current_report.get('face_shape', 'Bilinmiyor'),
                    self.current_report,
                    self.points_3d if hasattr(self, 'points_3d') else [],
                    landmarks_serializable,
                    side_image=side_img,
                    heatmap_image=heatmap_img
                )
                self.status_message.emit("✅ Analiz güncellendi ve veritabanına kaydedildi.")
                logging.info(f"Analysis updated in database, ID: {self.current_loaded_id}")
            else:
                # Save new record
                photo_id = db.save_analysis(
                    front_img,
                    self.current_report.get('face_shape', 'Bilinmiyor'),
                    self.current_report,
                    self.points_3d if hasattr(self, 'points_3d') else [],
                    landmarks_serializable,
                    side_image=side_img,
                    heatmap_image=heatmap_img
                )
                self.current_loaded_id = photo_id  # Store for potential updates
                self.status_message.emit(f"✅ Analiz kaydedildi (ID: {photo_id})")
                logging.info(f"Analysis saved to database, ID: {photo_id}")
            
            # Emit signal that save is complete
            print("DEBUG: Emitting analysis_saved signal")
            self.analysis_saved.emit()
                
        except Exception as e:
            logging.error(f"Kayıt hatası: {str(e)}", exc_info=True)
            self.status_message.emit(f"❌ Kayıt hatası: {str(e)}")

    def create_summary_tab(self):
        """Özet sekmesi içeriği (Zenginleştirilmiş)"""
        if self.summary_tab.layout():
            QWidget().setLayout(self.summary_tab.layout()) # Re-parent old layout
            
        layout = QHBoxLayout(self.summary_tab)
        layout.setContentsMargins(5, 5, 5, 5) # Reduced margins
        layout.setSpacing(10) # Reduced spacing
        
        # --- LEFT PANEL: Profile Card ---
        self.profile_card = QFrame()
        self.profile_card.setObjectName("profileCard")
        self.profile_card.setStyleSheet("""
            #profileCard {
                background-color: #1e1e2e;
                border: 1px solid #45475a;
                border-radius: 12px;
            }
        """)
        profile_layout = QVBoxLayout(self.profile_card)
        profile_layout.setContentsMargins(10, 15, 10, 15) # Reduced margins
        profile_layout.setSpacing(10) # Reduced spacing
        
        # 1. Header (Face Shape)
        lbl_title = QLabel("YÜZ PROFİLİ")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("color: #a6adc8; font-size: 11px; letter-spacing: 2px; font-weight: bold;")
        profile_layout.addWidget(lbl_title)
        
        self.shape_value = QLabel("-")
        self.shape_value.setAlignment(Qt.AlignCenter)
        self.shape_value.setStyleSheet("""
            color: #89b4fa;
            font-size: 24px; /* Reduced font size */
            font-weight: bold;
            padding: 8px;
            border: 2px solid #89b4fa;
            border-radius: 8px;
            background: rgba(137, 180, 250, 0.1);
        """)
        profile_layout.addWidget(self.shape_value)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #45475a;")
        profile_layout.addWidget(line)
        
        # 2. Dominant Traits (List)
        lbl_traits = QLabel("BASKIN ÖZELLİKLER")
        lbl_traits.setStyleSheet("color: #a6adc8; font-size: 10px; font-weight: bold; margin-top: 5px;")
        profile_layout.addWidget(lbl_traits)
        
        self.traits_list_layout = QVBoxLayout()
        self.traits_list_layout.setSpacing(5) # Reduced spacing
        profile_layout.addLayout(self.traits_list_layout)
        
        # Placeholder traits
        for _ in range(3):
            lbl = QLabel("• Analiz bekleniyor...")
            lbl.setStyleSheet("color: #cdd6f4; font-size: 12px;")
            self.traits_list_layout.addWidget(lbl)
            
        profile_layout.addStretch()
        
        profile_layout.addStretch()
        
        # 3. Footer (Energy/Score - Mockup) - REMOVED as per user request
        # footer_layout = QHBoxLayout()
        # lbl_energy = QLabel("Potansiyel:")
        # lbl_energy.setStyleSheet("color: #a6adc8; font-size: 11px;")
        # self.lbl_energy_val = QLabel("Yüksek")
        # self.lbl_energy_val.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 11px;")
        # footer_layout.addWidget(lbl_energy)
        # footer_layout.addWidget(self.lbl_energy_val)
        # footer_layout.addStretch()
        # profile_layout.addLayout(footer_layout)
        
        layout.addWidget(self.profile_card, stretch=3)
        
        # --- RIGHT PANEL: Attributes Grid ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        
        # Use Grid Layout for 2 columns
        self.attributes_grid = QGridLayout(content)
        self.attributes_grid.setContentsMargins(0, 0, 5, 0) # Reduced right margin
        self.attributes_grid.setSpacing(8) # Reduced spacing
        
        self.create_attribute_panels(self.attributes_grid)
        
        # Add stretch to push items up if needed, but grid handles it well
        # We can add a spacer at the bottom row
        self.attributes_grid.setRowStretch(10, 1) 
        
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=7)

    def create_attribute_panels(self, parent_layout):
        """Create attribute control panels (Grid Layout)"""
        
        # Helper to create styled group
        def create_group(title):
            group = QGroupBox(title)
            group.setStyleSheet("""
                QGroupBox {
                    font-weight: bold;
                    border: 1px solid #45475a;
                    border-radius: 6px;
                    margin-top: 8px; /* Reduced margin */
                    padding-top: 15px; /* Reduced padding */
                    background-color: #1e1e2e;
                    color: #cdd6f4;
                    font-size: 11px; /* Reduced font size */
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 8px;
                    padding: 0 3px;
                }
            """)
            return group

        # Helper for value labels
        def create_value_label():
            lbl = QLabel("-")
            lbl.setStyleSheet("color: #a6adc8; font-weight: normal; font-size: 11px;")
            return lbl
            
        # Helper for form layout
        def create_compact_form():
            form = QFormLayout()
            form.setContentsMargins(5, 5, 5, 5)
            form.setSpacing(4)
            form.setLabelAlignment(Qt.AlignLeft)
            return form

        # 1. Face Shape (Already in profile card, but keep detailed metrics here if any)
        # Let's skip Face Shape group here since it's prominent on the left
        # Or keep it for consistency? Let's keep it but maybe rename/refine.
        
        # 2. Forehead
        group_forehead = create_group("Alın")
        form = create_compact_form()
        self.lbl_forehead_width = create_value_label()
        self.lbl_forehead_height = create_value_label()
        self.lbl_forehead_slope = create_value_label()
        form.addRow("Genişlik:", self.lbl_forehead_width)
        form.addRow("Yükseklik:", self.lbl_forehead_height)
        form.addRow("Eğim:", self.lbl_forehead_slope)
        group_forehead.setLayout(form)
        parent_layout.addWidget(group_forehead, 0, 0)
        
        # 3. Eyes
        group_eyes = create_group("Gözler")
        form = create_compact_form()
        self.lbl_eyes_size = create_value_label()
        self.lbl_eyes_slant = create_value_label()
        self.lbl_eyes_spacing = create_value_label()
        self.lbl_eyes_depth = create_value_label()
        form.addRow("Büyüklük:", self.lbl_eyes_size)
        form.addRow("Eğim:", self.lbl_eyes_slant)
        form.addRow("Aralık:", self.lbl_eyes_spacing)
        form.addRow("Derinlik:", self.lbl_eyes_depth)
        group_eyes.setLayout(form)
        parent_layout.addWidget(group_eyes, 0, 1)
        
        # 4. Nose
        group_nose = create_group("Burun")
        form = create_compact_form()
        self.lbl_nose_length = create_value_label()
        self.lbl_nose_width = create_value_label()
        self.lbl_nose_bridge = create_value_label()
        self.lbl_nose_tip = create_value_label()
        form.addRow("Uzunluk:", self.lbl_nose_length)
        form.addRow("Genişlik:", self.lbl_nose_width)
        form.addRow("Kemer:", self.lbl_nose_bridge)
        form.addRow("Uç:", self.lbl_nose_tip)
        group_nose.setLayout(form)
        parent_layout.addWidget(group_nose, 1, 0)
        
        # 5. Lips
        group_lips = create_group("Dudaklar")
        form = create_compact_form()
        self.lbl_lips_upper = create_value_label()
        self.lbl_lips_lower = create_value_label()
        self.lbl_lips_width = create_value_label()
        form.addRow("Üst Kalınlık:", self.lbl_lips_upper)
        form.addRow("Alt Kalınlık:", self.lbl_lips_lower)
        form.addRow("Genişlik:", self.lbl_lips_width)
        group_lips.setLayout(form)
        parent_layout.addWidget(group_lips, 1, 1)
        
        # 6. Chin
        group_chin = create_group("Çene")
        form = create_compact_form()
        self.lbl_chin_width = create_value_label()
        self.lbl_chin_prominence = create_value_label()
        self.lbl_chin_dimple = create_value_label()
        form.addRow("Genişlik:", self.lbl_chin_width)
        form.addRow("Çıkıklık:", self.lbl_chin_prominence)
        form.addRow("Gamze:", self.lbl_chin_dimple)
        group_chin.setLayout(form)
        parent_layout.addWidget(group_chin, 2, 0)
        
        # 7. Ears
        group_ears = create_group("Kulaklar")
        form = create_compact_form()
        self.lbl_ears_size = create_value_label()
        self.lbl_ears_prominence = create_value_label()
        self.lbl_ears_lobe = create_value_label()
        form.addRow("Büyüklük:", self.lbl_ears_size)
        form.addRow("Kepçelik:", self.lbl_ears_prominence)
        form.addRow("Lob:", self.lbl_ears_lobe)
        group_ears.setLayout(form)
        parent_layout.addWidget(group_ears, 2, 1)
        
    def display_image(self, pixmap, label=None):
        """Resmi etikette göster (Orantılı ve Döngüsüz)"""
        if label is None:
            label = self.img_label
            
        if pixmap and not pixmap.isNull():
            # Store original pixmap for resizing
            label.original_pixmap = pixmap
            
            # Initial scale
            self.update_image_scaling(label)
        else:
            label.setText("Görüntü yok")
            if hasattr(label, 'original_pixmap'):
                del label.original_pixmap

    def update_image_scaling(self, label, force=False):
        """Helper to scale image to label size without growing loop"""
        if hasattr(label, 'original_pixmap') and label.original_pixmap:
            # Get current sizes
            target_size = label.size()
            pixmap_size = label.original_pixmap.size()
            
            # Don't scale if label is too small (not yet laid out)
            if target_size.width() < 10 or target_size.height() < 10:
                return
            
            # Don't scale if current pixmap already fits perfectly or is smaller
            # This prevents the growing loop
            current_pixmap = label.pixmap()
            if not force and current_pixmap and current_pixmap.size() == target_size:
                return
                
            # Scale to fit within the label while keeping aspect ratio
            scaled = label.original_pixmap.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            # Only set if it's actually different OR forced
            if force or not current_pixmap or scaled.size() != current_pixmap.size():
                label.setPixmap(scaled)


    def resizeEvent(self, event):
        """Pencere boyutu değiştiğinde resimleri yeniden ölçekle"""
        super().resizeEvent(event)
        # Update scaling for both labels
        if hasattr(self, 'img_label'):
            self.update_image_scaling(self.img_label)
        if hasattr(self, 'side_img_label'):
            self.update_image_scaling(self.side_img_label)

    def generate_annotated_image(self, clean_img, landmarks):
        """Helper to generate annotated image from clean image and landmarks"""
        try:
            from annotation_engine import AutoAnnotator
            from src.visualization import Visualizer
            
            # Convert clean numpy image to QPixmap
            rgb_image = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Prepare landmarks
            landmarks_obj = landmarks
            if isinstance(landmarks, list) and len(landmarks) > 0:
                if isinstance(landmarks[0], dict):
                    # Convert dicts to objects
                    class LandmarkObj:
                        def __init__(self, d):
                            self.x = d.get('x', 0)
                            self.y = d.get('y', 0)
                            self.z = d.get('z', 0)
                    landmarks_obj = [LandmarkObj(l) for l in landmarks]
            # Draw measurements on the pixmap
            annotator = AutoAnnotator()
            vis_pixmap = Visualizer.draw_measurements(pixmap, landmarks_obj, annotator)
            
            return vis_pixmap
            
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Failed to generate annotations: {e}")
            traceback.print_exc()
            return None

    def toggle_annotations(self, checked):
        """Toggle between clean and annotated images"""
        if not hasattr(self, 'clean_image') or self.clean_image is None:
            return
            
        # Update button text
        if checked:
            self.btn_toggle_annotations.setText("🙈 Etiketleri Gizle")
        else:
            self.btn_toggle_annotations.setText("👁️ Etiketleri Göster")
            
        target_pixmap = None
        
        if checked:
            # Show annotated image
            if hasattr(self, 'annotated_image') and self.annotated_image is not None:
                target_pixmap = self.annotated_image
                print(f"DEBUG: Using cached annotated_image. Size: {target_pixmap.width()}x{target_pixmap.height()}")
            else:
                # Fallback: Generate if missing (shouldn't happen with new flow)
                print("DEBUG: Annotated image missing, generating on fly...")
                target_pixmap = self.generate_annotated_image(self.clean_image, self.landmarks)
                self.annotated_image = target_pixmap # Cache it
                if target_pixmap:
                    print(f"DEBUG: Generated annotated_image. Size: {target_pixmap.width()}x{target_pixmap.height()}")
        else:
            # Show clean image
            # Convert numpy to QPixmap
            clean_img = self.clean_image
            rgb_image = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            target_pixmap = QPixmap.fromImage(q_img)
            print(f"DEBUG: Created clean pixmap. Size: {target_pixmap.width()}x{target_pixmap.height()}")
            
        if target_pixmap:
            # CRITICAL: Update original_pixmap so resizeEvent doesn't revert
            self.img_label.original_pixmap = target_pixmap
            
            # Use the centralized scaling method with FORCE update
            self.update_image_scaling(self.img_label, force=True)
        """Toggle between clean and annotated images"""
        print(f"DEBUG toggle_annotations called, checked={checked}")
        
        if not hasattr(self, 'clean_image') or self.clean_image is None:
            print("DEBUG: No clean_image available")
            return
        
        # Update button text
        if checked:
            self.btn_toggle_annotations.setText("🙈 Etiketleri Gizle")
            print("DEBUG: Button text set to 'Gizle'")
        else:
            self.btn_toggle_annotations.setText("👁️ Etiketleri Göster")
            print("DEBUG: Button text set to 'Göster'")
        
        # Get clean image (must be numpy array)
        clean_img = self.clean_image
        
        # Convert to numpy if it's QPixmap (shouldn't happen but safety check)
        if isinstance(clean_img, QPixmap):
            print("WARNING: clean_image is QPixmap, converting to numpy")
            qimg = clean_img.toImage()
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
            width = qimg.width()
            height = qimg.height()
            bytes_per_line = qimg.bytesPerLine()
            ptr = qimg.bits()
            arr = np.array(ptr).reshape(height, bytes_per_line)
            arr = arr[:, :width * 3].reshape(height, width, 3)
            clean_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        print(f"DEBUG: clean_img type={type(clean_img)}, shape={clean_img.shape if isinstance(clean_img, np.ndarray) else 'N/A'}")
        
        # If unchecked, show clean image
        if not checked:
            print("DEBUG: Showing CLEAN image (no annotations)")
            # Convert numpy to QPixmap
            rgb_image = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.display_image(pixmap, self.img_label)
            print("DEBUG: Clean image displayed")
            return
        
        # If checked, generate annotated image from CLEAN numpy image
        if not hasattr(self, 'landmarks') or self.landmarks is None:
            return
        
        try:
            from annotation_engine import AutoAnnotator
            from src.visualization import Visualizer
            
            # Convert clean numpy image to QPixmap
            rgb_image = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Prepare landmarks
            landmarks_obj = self.landmarks
            if isinstance(self.landmarks, list) and len(self.landmarks) > 0:
                if isinstance(self.landmarks[0], dict):
                    # Convert dicts to objects
                    class LandmarkObj:
                        def __init__(self, d):
                            self.x = d.get('x', 0)
                            self.y = d.get('y', 0)
                            self.z = d.get('z', 0)
                    landmarks_obj = [LandmarkObj(l) for l in self.landmarks]
            
            # Draw measurements on the pixmap
            annotator = AutoAnnotator()
            vis_pixmap = Visualizer.draw_measurements(pixmap, landmarks_obj, annotator)
            
            # Display the annotated image
            self.display_image(vis_pixmap, self.img_label)
            
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Failed to generate annotations: {e}")
            traceback.print_exc()

    def create_details_tab(self):
        """Detaylar sekmesi içeriği (Kişilik Karakteri Planı)"""
        if self.details_tab.layout():
             QWidget().setLayout(self.details_tab.layout())
             
        layout = QVBoxLayout(self.details_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        self.details_layout = QVBoxLayout(content)
        self.details_layout.setSpacing(15)
        self.details_layout.setAlignment(Qt.AlignTop)
        
        lbl_placeholder = QLabel("Analiz sonuçları bekleniyor...")
        lbl_placeholder.setAlignment(Qt.AlignCenter)
        lbl_placeholder.setStyleSheet("color: #6c7086; font-size: 14px; font-style: italic; margin-top: 20px;")
        self.details_layout.addWidget(lbl_placeholder)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def populate_details_tab(self, detailed_analysis):
        """Detaylı karakter analizi verilerini ekrana basar."""
        # Clear existing items
        while self.details_layout.count():
            item = self.details_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if not detailed_analysis or all(len(v) == 0 for v in detailed_analysis.values()):
            lbl = QLabel("Detaylı analiz verisi bulunamadı veya belirgin özellik tespit edilemedi.")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #a6adc8; font-style: italic; margin-top: 20px;")
            self.details_layout.addWidget(lbl)
            self.details_layout.addStretch()
            return
            
        colors = {
            "Zihin ve Bilişsel Yapı": "#89b4fa", # Blue
            "Sosyal ve İletişim": "#a6e3a1",     # Green
            "Duygu ve İrade": "#f9e2af",         # Yellow
            "Genel Karakter": "#cba6f7"          # Purple
        }
        
        for category, traits in detailed_analysis.items():
            if not traits:
                continue
                
            group = QGroupBox(category)
            color = colors.get(category, "#cdd6f4")
            group.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    border: 1px solid #45475a;
                    border-radius: 8px;
                    margin-top: 15px;
                    padding-top: 25px;
                    padding-left: 10px;
                    padding-right: 10px;
                    padding-bottom: 10px;
                    background-color: #181825;
                    color: {color};
                    font-size: 14px;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px;
                    padding: 0 5px;
                }}
            """)
            
            glayout = QVBoxLayout(group)
            glayout.setSpacing(8)
            
            text = " • ".join(traits)
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #cdd6f4; font-size: 13px; line-height: 1.5;")
            glayout.addWidget(lbl)
            
            self.details_layout.addWidget(group)
        
        self.details_layout.addStretch()

    def create_history_panel(self):
        """Geçmiş analizler paneli"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 0)
        
        # Header with toggle
        header_layout = QHBoxLayout()
        
        title = QLabel("📚 Son Analizler")
        title.setObjectName("sectionTitle")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.btn_toggle_history = QPushButton("▼")
        self.btn_toggle_history.setFixedSize(30, 30)
        self.btn_toggle_history.setCheckable(True)
        self.btn_toggle_history.setChecked(True)
        self.btn_toggle_history.clicked.connect(self.toggle_history)
        header_layout.addWidget(self.btn_toggle_history)
        
        layout.addLayout(header_layout)
        
        # History content (collapsible)
        self.history_content = QScrollArea()
        self.history_content.setFixedHeight(150)
        self.history_content.setWidgetResizable(True)
        self.history_content.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        self.history_layout = QHBoxLayout(content_widget)
        self.history_layout.setSpacing(10)
        self.history_layout.addStretch()
        
        self.history_content.setWidget(content_widget)
        layout.addWidget(self.history_content)
        
        return panel

    def toggle_history(self, checked):
        """Geçmiş panelini aç/kapat"""
        if checked:
            self.history_content.show()
            self.btn_toggle_history.setText("▼")
            self.refresh_history() # Refresh when opened
        else:
            self.history_content.hide()
            self.btn_toggle_history.setText("▲")

    def refresh_history(self):
        """Geçmiş listesini yenile (Veritabanından thumbnail oluştur)"""
        # Clear existing items
        while self.history_layout.count():
            item = self.history_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.spacerItem():
                pass
        
        from desktop_app.database import Database
        import cv2
        import numpy as np
        
        try:
            db = Database()
            analyses = db.get_recent_analyses()
            
            print(f"DEBUG refresh_history: {len(analyses)} kayıt bulundu")
            
            for row in analyses:
                photo_id = row['id']
                
                # Create card
                card = QPushButton()
                card.setFixedSize(120, 130)
                card.setStyleSheet("""
                    QPushButton {
                        background-color: #313244;
                        border: 1px solid #45475a;
                        border-radius: 8px;
                        text-align: center;
                    }
                    QPushButton:hover {
                        background-color: #45475a;
                        border: 1px solid #89b4fa;
                    }
                """)
                
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(5, 5, 5, 5)
                card_layout.setSpacing(2)
                
                # Thumbnail label
                img_label = QLabel()
                img_label.setFixedSize(100, 80)
                img_label.setStyleSheet("background-color: #1e1e2e; border-radius: 4px;")
                img_label.setAlignment(Qt.AlignCenter)
                
                # Get thumbnail from database
                try:
                    print(f"DEBUG: Loading thumbnail for ID {photo_id}")
                    # Fetch only front_image BLOB for thumbnail
                    import sqlite3
                    conn = sqlite3.connect(db.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT front_image FROM photos WHERE id = ?", (photo_id,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result and result[0]:
                        print(f"DEBUG: BLOB found, size: {len(result[0])} bytes")
                        # Decode BLOB to image
                        nparr = np.frombuffer(result[0], np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            print(f"DEBUG: Image decoded, shape: {img.shape}")
                            # Convert to QPixmap and scale
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            h, w, ch = rgb_img.shape
                            bytes_per_line = ch * w
                            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                            pixmap = QPixmap.fromImage(q_img)
                            scaled = pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            img_label.setPixmap(scaled)
                            print(f"DEBUG: Thumbnail set successfully")
                        else:
                            print(f"DEBUG: Image decode failed")
                            img_label.setText("Hata")
                    else:
                        print(f"DEBUG: No BLOB found for ID {photo_id}")
                        img_label.setText("Resim\nYok")
                except Exception as e:
                    print(f"DEBUG: Thumbnail load error for ID {photo_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    img_label.setText("Hata")
                
                card_layout.addWidget(img_label)
                
                # Date label
                date_str = row['timestamp'].split(' ')[0]
                lbl_date = QLabel(date_str)
                lbl_date.setAlignment(Qt.AlignCenter)
                lbl_date.setStyleSheet("color: #a6adc8; font-size: 10px;")
                card_layout.addWidget(lbl_date)
                
                # Face shape label
                lbl_shape = QLabel(row.get('face_shape') or "-")
                lbl_shape.setAlignment(Qt.AlignCenter)
                lbl_shape.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
                card_layout.addWidget(lbl_shape)
                
                # Click handler
                card.clicked.connect(lambda checked=False, r=row: self.load_analysis_from_history(r))
                
                self.history_layout.addWidget(card)
                
        except Exception as e:
            print(f"Geçmiş yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            err_label = QLabel("Geçmiş yüklenemedi")
            self.history_layout.addWidget(err_label)
            
        self.history_layout.addStretch()

    def load_analysis_from_history(self, row):
        """Veritabanından analiz yükle (row'dan ID al, DB'den tüm verileriçek)"""
        try:
            from desktop_app.database import Database
            import logging
            
            db = Database()
            photo_id = row['id']
            
            logging.info(f"Loading analysis from database, ID: {photo_id}")
            
            # Get full analysis data from database
            data = db.get_analysis_by_id(photo_id)
            
            if not data:
                logging.error(f"Analysis not found for ID: {photo_id}")
                self.status_message.emit(f"❌ Analiz bulunamadı (ID: {photo_id})")
                return
            
            # Extract data
            front_image = data['front_image']  # numpy array
            side_image = data['side_image']    # numpy array or None
            heatmap_image = data['heatmap_image']  # numpy array or None
            analysis = data.get('analysis', {})
            landmarks = data.get('landmarks', [])
            points_3d = data.get('points_3d', [])
            
            if front_image is None:
                logging.error("Front image is None")
                self.status_message.emit("❌ Ön görüntü yüklenemedi")
                return
            
            logging.info(f"Loaded: front:{front_image.shape if front_image is not None else None}, "
                        f"landmarks:{len(landmarks) if landmarks else 0}")
            
            # Store current loaded ID for updates
            self.current_loaded_id = photo_id
            
            # Display results
            # Since we have the clean front image and landmarks, 
            # we can generate the annotated version on demand via toggle
            self.display_results(
                report=analysis,
                visualized_image=heatmap_image if heatmap_image is not None else front_image,
                points_3d=np.array(points_3d) if points_3d else None,
                landmarks=landmarks,
                side_image=side_image,
                clean_image=front_image
            )
            
            # Store current loaded ID for updates (Must be set AFTER display_results)
            self.current_loaded_id = photo_id
            
            # Update button state
            self.btn_save.setText("💾 Güncelle")
            self.btn_save.setEnabled(True)
            
            self.status_message.emit(f"✅ Analiz yüklendi (ID: {photo_id})")
            logging.info(f"Analysis loaded successfully, ID: {photo_id}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Load failed: {e}")
            self.status_message.emit(f"❌ Yükleme hatası: {e}")
    
    def open_feedback(self):
        """Open feedback dialog"""
        if not hasattr(self, 'current_report') or not self.current_report:
            return
            
        # We need the original image path if available, or just pass the image data
        image_path = getattr(self, 'current_image_path', None)
        
        if not image_path:
            # Auto-save to dataset/raw if not exists (for live capture)
            from datetime import datetime
            import os
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            raw_dir = os.path.join(base_dir, "dataset", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            image_path = os.path.join(raw_dir, filename)
            
            if hasattr(self, 'current_image') and self.current_image is not None:
                cv2.imwrite(image_path, self.current_image)
                self.current_image_path = image_path
            else:
                return

        dialog = FeedbackDialog(self, self.current_image, self.current_report, image_path)
        dialog.exec()

    def display_results(self, report, visualized_image, points_3d, landmarks, side_image=None, clean_image=None):
        """Analiz sonuçlarını göster"""
        logging.info(f"DEBUG: display_results called. Clean image provided: {clean_image is not None}")
        if isinstance(clean_image, np.ndarray):
            logging.info(f"DEBUG: Clean image shape: {clean_image.shape}")
            
        self.current_report = report
        self.current_image = visualized_image
        
        # CRITICAL: Always store clean_image as numpy array, NEVER as QPixmap
        if clean_image is not None:
            if isinstance(clean_image, QPixmap):
                # Convert QPixmap to numpy
                qimg = clean_image.toImage()
                qimg = qimg.convertToFormat(QImage.Format_RGB888)
                width = qimg.width()
                height = qimg.height()
                bytes_per_line = qimg.bytesPerLine()
                ptr = qimg.bits()
                arr = np.array(ptr).reshape(height, bytes_per_line)
                arr = arr[:, :width * 3].reshape(height, width, 3)
                self.clean_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR).copy()
            else:
                self.clean_image = clean_image.copy() if hasattr(clean_image, 'copy') else clean_image
        else:
            self.clean_image = None
            
        # Generate and store annotated image immediately
        if self.clean_image is not None and self.landmarks:
            print("DEBUG: Pre-generating annotated image in display_results")
            self.annotated_image = self.generate_annotated_image(self.clean_image, self.landmarks)
        else:
            self.annotated_image = None
            
        self.side_image = side_image
        self.points_3d = points_3d
        self.landmarks = landmarks
        self.current_heatmap_image = visualized_image
        self.current_heatmap_image = visualized_image
        # self.current_loaded_id = None # REMOVED: Do not reset here, handled by caller
        
        # Reset Save button text to "Save"
        self.btn_save.setText("💾 Kaydet")
        
        # 1. Display Images
        # Reset toggle button to unchecked (Clean view by default)
        self.btn_toggle_annotations.blockSignals(True)
        self.btn_toggle_annotations.setChecked(False)
        self.btn_toggle_annotations.setText("👁️ Etiketleri Göster")
        self.btn_toggle_annotations.blockSignals(False)
        
        # Show CLEAN image by default
        display_img = clean_image if clean_image is not None else visualized_image
        
        if isinstance(display_img, QPixmap):
             pixmap = display_img
        else:
            # Convert BGR to RGB for Qt
            rgb_image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
        # CRITICAL: Update original_pixmap so resizeEvent doesn't revert
        self.img_label.original_pixmap = pixmap
        
        # Use the centralized scaling method with FORCE update
        # Defer this slightly to allow layout to settle (fixes "too big" initial image)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(10, lambda: self.update_image_scaling(self.img_label, force=True))
        
        if side_image is not None:
            rgb_side = cv2.cvtColor(side_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_side.shape
            bytes_per_line = ch * w
            q_side = QImage(rgb_side.data, w, h, bytes_per_line, QImage.Format_RGB888)
            side_pixmap = QPixmap.fromImage(q_side)
            self.current_side_pixmap = side_pixmap
            # Ön profil gibi ortalanmış gösterim
            self.side_img_label.original_pixmap = side_pixmap
            QTimer.singleShot(10, lambda: self.update_image_scaling(self.side_img_label, force=True))
        else:
            self.side_img_label.setText("Yan profil yok")
            self.current_side_pixmap = None
            
        # 2. Update 3D View
        if points_3d is not None and len(points_3d) > 0:
            self.lbl_3d_warning.hide()
            self.view_3d.show()
            self.controls_3d_widget.show()
            
            # Clear old items
            self.view_3d.items = []
            
            try:
                from scipy.spatial import Delaunay
                
                pos = np.array(points_3d, dtype=np.float32)
                
                # Center the mesh
                center = np.mean(pos, axis=0)
                pos = pos - center
                
                # Flip Y axis (MediaPipe Y is inverted)
                pos[:, 1] = -pos[:, 1]
                
                # Auto-scale: normalize to fit nicely in view
                bbox_size = pos.max(axis=0) - pos.min(axis=0)
                max_extent = max(bbox_size[0], bbox_size[1])
                if max_extent > 0:
                    scale_factor = 80.0 / max_extent  # Fit in ~80 unit box
                    pos = pos * scale_factor
                
                # Delaunay triangulation on X,Y projection
                points_2d = pos[:, :2]
                tri = Delaunay(points_2d)
                faces = tri.simplices.astype(np.uint32)
                
                # Filter out overly large triangles (artifacts at face edge)
                if len(faces) > 0:
                    # Calculate edge lengths for each triangle
                    valid_mask = np.ones(len(faces), dtype=bool)
                    max_edge_threshold = max_extent * scale_factor * 0.15  # %15 of face size
                    
                    for i, face in enumerate(faces):
                        v0, v1, v2 = pos[face[0]], pos[face[1]], pos[face[2]]
                        e1 = np.linalg.norm(v1 - v0)
                        e2 = np.linalg.norm(v2 - v1)
                        e3 = np.linalg.norm(v0 - v2)
                        if max(e1, e2, e3) > max_edge_threshold:
                            valid_mask[i] = False
                    
                    faces = faces[valid_mask]
                
                if len(faces) > 0:
                    # Color gradient based on Z depth
                    z = pos[:, 2]
                    z_diff = z.max() - z.min()
                    z_norm = (z - z.min()) / z_diff if z_diff > 0 else np.zeros_like(z)
                    
                    # Skin-like color palette (warm tones)
                    colors = np.zeros((len(pos), 4), dtype=np.float32)
                    colors[:, 0] = 0.55 + z_norm * 0.25  # R: warm base
                    colors[:, 1] = 0.42 + z_norm * 0.2   # G: subtle green
                    colors[:, 2] = 0.35 + z_norm * 0.15  # B: less blue
                    colors[:, 3] = 0.92                    # Alpha
                    
                    mesh = gl.GLMeshItem(
                        vertexes=pos,
                        faces=faces,
                        vertexColors=colors,
                        smooth=True,
                        drawEdges=False,
                        drawFaces=True,
                        shader='shaded',
                        glOptions='translucent'
                    )
                    self.view_3d.addItem(mesh)
                    
                    # Subtle wireframe
                    wireframe = gl.GLMeshItem(
                        vertexes=pos,
                        faces=faces,
                        drawEdges=True,
                        drawFaces=False,
                        edgeColor=(0.6, 0.5, 0.4, 0.08),
                        smooth=False,
                        glOptions='translucent'
                    )
                    self.view_3d.addItem(wireframe)
                else:
                    sp = gl.GLScatterPlotItem(pos=pos, color=(0.55, 0.42, 0.35, 0.8), size=2, pxMode=True)
                    self.view_3d.addItem(sp)
                    
            except Exception as e:
                logging.error(f"3D Mesh oluşturma hatası: {e}")
                import traceback
                traceback.print_exc()
                pos = np.array(points_3d, dtype=np.float32)
                center = np.mean(pos, axis=0)
                pos = pos - center
                pos[:, 1] = -pos[:, 1]
                bbox_size = pos.max(axis=0) - pos.min(axis=0)
                max_ext = max(bbox_size[0], bbox_size[1])
                if max_ext > 0:
                    pos = pos * (80.0 / max_ext)
                sp = gl.GLScatterPlotItem(pos=pos, color=(0.55, 0.42, 0.35, 0.8), size=2, pxMode=True)
                self.view_3d.addItem(sp)
            
            # Camera: face-centered view
            self.view_3d.setCameraPosition(distance=180, elevation=5, azimuth=0)
        else:
            self.view_3d.hide()
            self.controls_3d_widget.hide()
            self.lbl_3d_warning.show()
        
        # 3. Update Summary Tab
        self.update_profile_card(report)
        
        # 4. Populate Detailed Traits Tab
        detailed_analysis = report.get('detailed_analysis', {})
        self.populate_details_tab(detailed_analysis)
        
        # 4. Update Attribute Panels (Populate labels)
        # Extract annotations from report if available
        annotations = report.get('annotations', {})

        # Forehead
        fh = annotations.get('forehead', {})
        self.lbl_forehead_width.setText(str(fh.get('width', '-')))
        self.lbl_forehead_height.setText(str(fh.get('height', '-')))
        self.lbl_forehead_slope.setText(str(fh.get('slope', '-')))
        
        # Eyes
        eyes = annotations.get('eyes', {})
        self.lbl_eyes_size.setText(str(eyes.get('size', '-')))
        self.lbl_eyes_slant.setText(str(eyes.get('slant', '-')))
        self.lbl_eyes_spacing.setText(str(eyes.get('spacing', '-')))
        self.lbl_eyes_depth.setText(str(eyes.get('depth', '-')))
        
        # Nose
        nose = annotations.get('nose', {})
        self.lbl_nose_length.setText(str(nose.get('length', '-')))
        self.lbl_nose_width.setText(str(nose.get('width', '-')))
        self.lbl_nose_bridge.setText(str(nose.get('bridge', '-')))
        self.lbl_nose_tip.setText(str(nose.get('tip', '-')))
        
        # Lips
        lips = annotations.get('lips', {})
        self.lbl_lips_upper.setText(str(lips.get('upper_thickness', '-')))
        self.lbl_lips_lower.setText(str(lips.get('lower_thickness', '-')))
        self.lbl_lips_width.setText(str(lips.get('width', '-')))
        
        # Chin
        chin = annotations.get('chin', {})
        self.lbl_chin_width.setText(str(chin.get('width', '-')))
        self.lbl_chin_prominence.setText(str(chin.get('prominence', '-')))
        self.lbl_chin_dimple.setText(str(chin.get('dimple', '-')))
        
        # Ears
        ears = annotations.get('ears', {})
        self.lbl_ears_size.setText(str(ears.get('size', '-')))
        self.lbl_ears_prominence.setText(str(ears.get('prominence', '-')))
        self.lbl_ears_lobe.setText(str(ears.get('lobe', '-')))

        # Update Radar Chart (Removed)
        # self.update_radar_chart(report)
        
        # Display trait cards
        # 5. Update Attribute Panels (Right Side) - Already done above
        
        # 6. Display Trait Cards (Removed for now as Details tab is placeholder)
        # self.display_trait_cards(report)

    def update_profile_card(self, report):
        """Profil kartını güncelle"""
        # Clear existing traits
        while self.traits_list_layout.count():
            item = self.traits_list_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        # Set Face Shape
        # Try to get from annotations first, then report root
        shape = report.get('annotations', {}).get('face_shape', {}).get('shape')
        if not shape:
            shape = report.get('face_shape', '-')
        self.shape_value.setText(str(shape))
            
        # Get top traits
        analysis = report.get('analysis', {})
        top_traits = []
        
        if isinstance(analysis, dict):
            # Collect all traits
            all_t = []
            if analysis.get('positive'): all_t.extend(analysis['positive'])
            if analysis.get('negative'): all_t.extend(analysis['negative'])
            
            # Pick top 3-5
            for t in all_t[:5]:
                if isinstance(t, dict):
                    trait_text = t.get('trait', '')
                else:
                    trait_text = str(t)
                
                # Clean up trait text (remove physical description after :)
                if ':' in trait_text:
                    trait_text = trait_text.split(':')[0].strip()
                    
                top_traits.append(trait_text)
                    
        if not top_traits:
            top_traits = ["Analiz bekleniyor..."]
            
        for trait in top_traits:
            lbl = QLabel(f"• {trait}")
            lbl.setStyleSheet("color: #cdd6f4; font-size: 13px;")
            lbl.setWordWrap(True)
            self.traits_list_layout.addWidget(lbl)



    def create_heatmap(self, image, report, landmarks=None):
        """
        Yenilenmiş Isı Haritası ve Yüz Ortalama Mantığı.
        Face Centering & Heatmap Generation.
        """
        if landmarks is None:
            self.img_label.setText("Landmarks bulunamadı.")
            return

        # 1. Prepare Data
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Convert landmarks to pixel coordinates
        if hasattr(landmarks, 'landmark'):
            lm_list = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            landmarks_array = np.array(lm_list)
        elif isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], dict):
            lm_list = [[lm['x'], lm['y'], lm['z']] for lm in landmarks]
            landmarks_array = np.array(lm_list)
        else:
            landmarks_array = np.array(landmarks)
            
        if np.max(landmarks_array) <= 1.0:
            pixel_landmarks = landmarks_array[:, :2] * np.array([w, h])
        else:
            pixel_landmarks = landmarks_array[:, :2]
            
        pixel_landmarks = pixel_landmarks.astype(np.int32)

        # 2. Draw Heatmap Zones (On Full Image)
        # Zone Indices (MediaPipe 468)
        ZONE_INDICES = {
            'Alın': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'Sol Göz': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            'Sağ Göz': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            'Burun': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64, 98],
            'Dudaklar': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
            'Çene': [175, 199, 200, 18, 201, 8, 428, 262, 431, 418, 421, 200],
            'Sol Kaş': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'Sağ Kaş': [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
            'Sol Kulak': [127, 234, 93, 132, 58, 172, 136, 150],
            'Sağ Kulak': [356, 454, 323, 361, 288, 397, 365, 379],
            'Sol Yanak': [101, 50, 123, 137, 177, 215, 138, 135, 169, 170, 140, 171, 32, 195, 5, 4, 111, 117, 118, 100, 126, 209, 49],
            'Sağ Yanak': [330, 280, 352, 366, 401, 435, 367, 364, 394, 395, 369, 396, 262, 419, 6, 197, 195, 5, 4, 1, 19, 94, 2]
        }

        heatmap_layer = np.zeros_like(image)
        
        for zone_name, indices in ZONE_INDICES.items():
            # Get points for this zone
            points = pixel_landmarks[indices]
            
            # Determine Color
            color = (0, 0, 255) # Default Red
            if zone_name == 'Alın': color = (0, 255, 255) # Yellow
            elif zone_name == 'Burun': color = (0, 165, 255) # Orange
            elif zone_name == 'Dudaklar': color = (0, 0, 255) # Red
            elif zone_name == 'Çene': color = (0, 255, 0) # Green
            elif 'Kaş' in zone_name: color = (255, 0, 255) # Magenta
            elif 'Göz' in zone_name: color = (255, 255, 0) # Cyan
            elif 'Kulak' in zone_name: color = (255, 165, 0) 
            elif 'Yanak' in zone_name: color = (128, 0, 128) # Purple
            
            # Draw filled polygon on heatmap layer
            hull = cv2.convexHull(points)
            cv2.fillPoly(heatmap_layer, [hull], color)
            
            # Add Text Label (Turkish)
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Draw text with outline for visibility
                self.draw_turkish_text(heatmap_layer, zone_name, (cX-20, cY))

        # Blend Heatmap with Original
        alpha = 0.35
        cv2.addWeighted(heatmap_layer, alpha, overlay, 1 - alpha, 0, overlay)

        # 3. Robust Centering & Cropping
        # Calculate Face Bounding Box
        x_min, y_min = np.min(pixel_landmarks, axis=0)
        x_max, y_max = np.max(pixel_landmarks, axis=0)
        
        face_w = x_max - x_min
        face_h = y_max - y_min
        
        # Determine Center
        center_x = x_min + face_w / 2
        center_y = y_min + face_h / 2
        
        # Determine Crop Size (Square)
        # We want the face to occupy about 60% of the frame
        # So frame_size = max_face_dim / 0.6
        max_dim = max(face_w, face_h)
        crop_size = int(max_dim * 1.8) # 1.8x padding
        
        # Calculate Crop Coordinates (top-left)
        x1 = int(center_x - crop_size / 2)
        y1 = int(center_y - crop_size / 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        # Handle Boundaries (Pad if necessary)
        # Create a black canvas large enough to hold the crop
        canvas_size = crop_size
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Calculate intersection with original image
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(w, x2)
        src_y2 = min(h, y2)
        
        dst_x1 = max(0, src_x1 - x1)
        dst_y1 = max(0, src_y1 - y1)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Copy valid region to canvas
        if src_x2 > src_x1 and src_y2 > src_y1:
            # Note: overlay is the image with heatmap drawn on it
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = overlay[src_y1:src_y2, src_x1:src_x2]
            
        self.current_heatmap_image = canvas
        self.display_image(canvas, self.img_label)

    
    def create_3d_mesh(self, points):
        """3D mesh oluştur"""
        self.view_3d.clear()
        
        # Convert points to centered coordinates
        points_centered = points.copy()
        center = points_centered.mean(axis=0)
        points_centered -= center
        
        # Create point cloud
        scatter = gl.GLScatterPlotItem(
            pos=points_centered,
            color=(0.5, 0.7, 1.0, 0.8),
            size=3,
            pxMode=True
        )
        self.view_3d.addItem(scatter)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.scale(50, 50, 50)
        grid.translate(0, 0, -100)
        self.view_3d.addItem(grid)
    
    def update_radar_chart(self, report):
        """Radar Chart güncelle"""
        self.radar_plot.clear()
        # Mock data for now
        categories = ['Zeka', 'Duygu', 'İrade', 'Enerji', 'Sosyal']
        values = [0.8, 0.6, 0.7, 0.9, 0.5] # This should come from report
        
        # Draw radar chart (Polar plot)
        # ... (Implementation detail for simple polar plot)
        # For simplicity, let's just use a bar chart for now or simple text if radar is too complex for quick edit
        # Let's stick to simple text summary in the plot area for now to avoid complex math errors
        text = pg.TextItem(text="Analiz Profili\n\nZeka: 80%\nDuygu: 60%\nİrade: 70%", color=(200, 200, 200), anchor=(0.5, 0.5))
        self.radar_plot.addItem(text)

    def display_trait_cards(self, report):
        """Grid layout ile kartları göster"""
        # Clear grid
        while self.traits_grid.count():
            item = self.traits_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        row, col = 0, 0
        
        all_traits = []
        # Safely get analysis lists
        analysis = report.get('analysis', {})
        if isinstance(analysis, str):
             # Handle case where analysis might be a string (error case)
             print(f"Analysis is string: {analysis}")
             return

        if analysis.get('positive'):
            for t in analysis['positive']: all_traits.append((t, "positive"))
        if analysis.get('negative'):
            for t in analysis['negative']: all_traits.append((t, "negative"))
        if analysis.get('neutral'):
            for t in analysis['neutral']: all_traits.append((t, "neutral"))
            
        print(f"DEBUG: Found {len(all_traits)} traits to display.")
            
        for trait_data, category in all_traits:
            # trait_data can be dict or string
            if isinstance(trait_data, dict):
                title = trait_data.get('trait', 'Özellik')
                desc = trait_data.get('description', '')
            else:
                title = str(trait_data)
                desc = ""
                
            card = self.create_trait_card(title, desc, category)
            self.traits_grid.addWidget(card, row, col)
            
            col += 1
            if col > 1: # 2 columns
                col = 0
                row += 1
                
        # Force update
        self.details_tab.update()

    def create_trait_card(self, title, desc, category):
        card = QFrame()
        colors = {"positive": "#313244", "negative": "#313244", "neutral": "#313244"}
        border_colors = {"positive": "#a6e3a1", "negative": "#f38ba8", "neutral": "#89b4fa"}
        
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {colors.get(category, '#313244')};
                border-left: 4px solid {border_colors.get(category, '#89b4fa')};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        
        l = QVBoxLayout(card)
        l.setContentsMargins(5, 5, 5, 5)
        l.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setWordWrap(True)
        lbl_title.setStyleSheet("font-weight: bold; color: white; font-size: 13px;")
        l.addWidget(lbl_title)
        
        if desc:
            lbl_desc = QLabel(desc)
            lbl_desc.setWordWrap(True)
            lbl_desc.setStyleSheet("color: #cdd6f4; font-size: 11px;")
            l.addWidget(lbl_desc)
            
        l.addStretch()
        return card
    
    def toggle_rotation(self, checked):
        """3D otomatik rotasyonu aç/kapat"""
        if checked:
            self.rotation_timer.start(50)  # 20 FPS
            self.btn_rotate.setText("⏸ Durakla")
        else:
            self.rotation_timer.stop()
            self.btn_rotate.setText("🔄 Otomatik Döndür")
    
    def rotate_3d(self):
        """3D modeli döndür"""
        self.rotation_angle += 2
        self.view_3d.setCameraPosition(
            azimuth=self.rotation_angle,
            distance=500
        )
    
    def draw_turkish_text(self, img, text, pos):
        """PIL kullanarak Türkçe karakter destekli metin çiz"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font that supports Turkish
        try:
            if platform.system() == "Darwin": # macOS
                font = ImageFont.truetype("Arial.ttf", 16)
            elif platform.system() == "Windows":
                font = ImageFont.truetype("arial.ttf", 16)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        # Draw outline
        x, y = pos
        outline_color = (0, 0, 0)
        text_color = (255, 255, 255)
        
        # Thick outline
        for adj in range(-2, 3):
            for adj2 in range(-2, 3):
                draw.text((x+adj, y+adj2), text, font=font, fill=outline_color)
                
        # Text
        draw.text((x, y), text, font=font, fill=text_color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def reset_3d_view(self):
        """3D görünümü sıfırla"""
        self.rotation_angle = 0
        self.view_3d.setCameraPosition(
            azimuth=0,
            elevation=20,
            distance=500
        )
        if self.btn_rotate.isChecked():
            self.btn_rotate.click()
