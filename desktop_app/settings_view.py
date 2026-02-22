from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QCheckBox, QPushButton, QFrame,
    QGroupBox, QApplication, QTabWidget, QSlider, QSpinBox
)
from PySide6.QtCore import Qt, Signal

class SettingsView(QWidget):
    """Ayarlar görünümü"""
    
    # Signal to notify main window
    settings_saved = Signal(str) # message
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("⚙️ Ayarlar")
        title.setObjectName("viewTitle")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(title)
        
        # Create Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 6px;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background-color: #45475a;
                color: #89b4fa;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #3d3f4e;
            }
        """)
        
        # Create tabs
        self.tab_application = self.create_application_tab()
        self.tab_analysis = self.create_analysis_tab()
        self.tab_appearance = self.create_appearance_tab()
        self.tab_about = self.create_about_tab()
        
        # Add tabs
        self.tab_widget.addTab(self.tab_application, "📱 Uygulama")
        self.tab_widget.addTab(self.tab_analysis, "🔬 Analiz")
        self.tab_widget.addTab(self.tab_appearance, "🎨 Görünüm")
        self.tab_widget.addTab(self.tab_about, "ℹ️ Hakkında")
        
        layout.addWidget(self.tab_widget)
        
        # Save Button (Bottom)
        self.btn_save = QPushButton("💾 Ayarları Kaydet")
        self.btn_save.setFixedSize(200, 45)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                border-radius: 6px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #b4befe;
            }
        """)
        self.btn_save.clicked.connect(self.save_settings)
        layout.addWidget(self.btn_save, alignment=Qt.AlignCenter)
    
    def create_application_tab(self):
        """Uygulama ayarları tab'ı"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Pre-Processing Group
        group_pre = QGroupBox("🔧 Ön İşleme (Pre-Processing)")
        group_pre.setStyleSheet(self.get_group_style())
        pre_layout = QVBoxLayout(group_pre)
        
        # Image auto-enhance
        self.chk_auto_enhance = QCheckBox("Görüntüyü otomatik iyileştir")
        self.chk_auto_enhance.setChecked(True)
        pre_layout.addWidget(self.chk_auto_enhance)
        
        # Auto brightness
        self.chk_auto_brightness = QCheckBox("Parlaklığı otomatik ayarla")
        self.chk_auto_brightness.setChecked(False)
        pre_layout.addWidget(self.chk_auto_brightness)
        
        # Noise reduction
        noise_layout = QHBoxLayout()
        noise_layout.addWidget(QLabel("Gürültü azaltma:"))
        self.slider_noise = QSlider(Qt.Horizontal)
        self.slider_noise.setRange(0, 10)
        self.slider_noise.setValue(5)
        noise_layout.addWidget(self.slider_noise)
        self.lbl_noise_val = QLabel("5")
        self.slider_noise.valueChanged.connect(lambda v: self.lbl_noise_val.setText(str(v)))
        noise_layout.addWidget(self.lbl_noise_val)
        pre_layout.addLayout(noise_layout)
        
        # Camera Resolution
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("📷 Kamera çözünürlüğü:"))
        self.combo_camera_res = QComboBox()
        self.combo_camera_res.addItems(["Düşük (640x480)", "Orta (1280x720)", "Yüksek (1920x1080)"])
        self.combo_camera_res.setCurrentIndex(1)  # Default: Medium
        res_layout.addWidget(self.combo_camera_res)
        res_layout.addStretch()
        pre_layout.addLayout(res_layout)
        
        # Mesh Orientation Section
        pre_layout.addWidget(QLabel("🔧 Mesh Orientation:"))
        
        # Distance
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Distance:"))
        self.spin_mesh_distance = QSpinBox()
        self.spin_mesh_distance.setRange(200, 800)
        self.spin_mesh_distance.setValue(400)
        dist_layout.addWidget(self.spin_mesh_distance)
        dist_layout.addStretch()
        pre_layout.addLayout(dist_layout)
        
        # Elevation
        elev_layout = QHBoxLayout()
        elev_layout.addWidget(QLabel("Elevation (°):"))
        self.spin_mesh_elevation = QSpinBox()
        self.spin_mesh_elevation.setRange(0, 90)
        self.spin_mesh_elevation.setValue(70)
        elev_layout.addWidget(self.spin_mesh_elevation)
        elev_layout.addStretch()
        pre_layout.addLayout(elev_layout)
        
        # Azimuth
        azi_layout = QHBoxLayout()
        azi_layout.addWidget(QLabel("Azimuth (°):"))
        self.spin_mesh_azimuth = QSpinBox()
        self.spin_mesh_azimuth.setRange(0, 360)
        self.spin_mesh_azimuth.setValue(0)
        azi_layout.addWidget(self.spin_mesh_azimuth)
        azi_layout.addStretch()
        pre_layout.addLayout(azi_layout)
        
        layout.addWidget(group_pre)
        
        # Post-Processing Group
        group_post = QGroupBox("✨ Son İşleme (Post-Processing)")
        group_post.setStyleSheet(self.get_group_style())
        post_layout = QVBoxLayout(group_post)
        
        # Smoothing
        self.chk_smoothing = QCheckBox("Mesh yumuşatma uygula")
        self.chk_smoothing.setChecked(True)
        post_layout.addWidget(self.chk_smoothing)
        
        # Detail level
        detail_layout = QHBoxLayout()
        detail_layout.addWidget(QLabel("Detay seviyesi:"))
        self.combo_detail = QComboBox()
        self.combo_detail.addItems(["Düşük", "Orta", "Yüksek"])
        self.combo_detail.setCurrentIndex(1)
        detail_layout.addWidget(self.combo_detail)
        detail_layout.addStretch()
        post_layout.addLayout(detail_layout)
        
        # Auto-color correction
        self.chk_color_correct = QCheckBox("Renk düzeltme uygula")
        self.chk_color_correct.setChecked(False)
        post_layout.addWidget(self.chk_color_correct)
        
        layout.addWidget(group_post)
        
        # Auto-retry on low quality
        # self.chk_auto_retry = QCheckBox("Düşük kalitede otomatik yeniden dene")
        # self.chk_auto_retry.setChecked(False)
        # quality_layout.addWidget(self.chk_auto_retry)
        
        # Archive Refresh Interval
        refresh_layout = QHBoxLayout()
        refresh_layout.addWidget(QLabel("Arşiv yenileme süresi (sn):"))
        self.spin_refresh_interval = QSpinBox()
        self.spin_refresh_interval.setRange(0, 3600) # 0 = disabled
        self.spin_refresh_interval.setValue(0)
        self.spin_refresh_interval.setSpecialValueText("Kapalı")
        refresh_layout.addWidget(self.spin_refresh_interval)
        refresh_layout.addStretch()
        layout.addWidget(QLabel("⏱️ Otomasyon"))
        layout.addLayout(refresh_layout)
        
        layout.addStretch()
        return tab
    
    def create_analysis_tab(self):
        """Analiz ayarları tab'ı"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Analysis Settings Group
        group_analysis = QGroupBox("🔬 Analiz Ayarları")
        group_analysis.setStyleSheet(self.get_group_style())
        ana_layout = QVBoxLayout(group_analysis)
        
        # High precision
        self.chk_precision = QCheckBox("Yüksek hassasiyetli tarama (Daha yavaş)")
        ana_layout.addWidget(self.chk_precision)
        
        # Landmark count
        landmark_layout = QHBoxLayout()
        landmark_layout.addWidget(QLabel("Landmark sayısı:"))
        self.combo_landmarks = QComboBox()
        self.combo_landmarks.addItems(["468 (Standart)", "478 (Rafine)"])
        self.combo_landmarks.setCurrentIndex(1)
        landmark_layout.addWidget(self.combo_landmarks)
        landmark_layout.addStretch()
        ana_layout.addLayout(landmark_layout)
        
        # Min detection confidence
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Tespit güveni eşiği:"))
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(30, 90)
        self.spin_confidence.setValue(50)
        self.spin_confidence.setSuffix("%")
        conf_layout.addWidget(self.spin_confidence)
        conf_layout.addStretch()
        ana_layout.addLayout(conf_layout)
        
        # Multi-face support
        self.chk_multi_face = QCheckBox("Çoklu yüz desteği (sadece ilk yüz kullanılır)")
        self.chk_multi_face.setChecked(False)
        ana_layout.addWidget(self.chk_multi_face)
        
        layout.addWidget(group_analysis)
        
        # Quality Control Group
        group_quality = QGroupBox("📊 Kalite Kontrolü")
        group_quality.setStyleSheet(self.get_group_style())
        quality_layout = QVBoxLayout(group_quality)
        
        # Min quality threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Minimum mesh kalitesi:"))
        self.spin_quality_threshold = QSpinBox()
        self.spin_quality_threshold.setRange(30, 90)
        self.spin_quality_threshold.setValue(60)
        self.spin_quality_threshold.setSuffix("%")
        threshold_layout.addWidget(self.spin_quality_threshold)
        threshold_layout.addStretch()
        quality_layout.addLayout(threshold_layout)
        
        # Auto-retry on low quality
        self.chk_auto_retry = QCheckBox("Düşük kalitede otomatik yeniden dene")
        self.chk_auto_retry.setChecked(False)
        quality_layout.addWidget(self.chk_auto_retry)
        
        layout.addWidget(group_quality)
        
        layout.addStretch()
        return tab
    
    def create_appearance_tab(self):
        """Görünüm tab'ı"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Appearance Group
        group_appearance = QGroupBox("🎨 Görünüm")
        group_appearance.setStyleSheet(self.get_group_style())
        app_layout = QVBoxLayout(group_appearance)
        
        # Theme
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Tema:"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["Koyu (Varsayılan)", "Açık", "Sistem"])
        theme_layout.addWidget(self.combo_theme)
        theme_layout.addStretch()
        app_layout.addLayout(theme_layout)
        
        # Font size
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Yazı boyutu:"))
        self.combo_font_size = QComboBox()
        self.combo_font_size.addItems(["Küçük", "Orta (Varsayılan)", "Büyük"])
        self.combo_font_size.setCurrentIndex(1)
        font_layout.addWidget(self.combo_font_size)
        font_layout.addStretch()
        app_layout.addLayout(font_layout)
        
        # Show grid
        self.chk_show_grid = QCheckBox("Mesh'te grid göster")
        self.chk_show_grid.setChecked(True)
        app_layout.addWidget(self.chk_show_grid)
        
        layout.addWidget(group_appearance)
        
        layout.addStretch()
        return tab
    
    def create_about_tab(self):
        """Hakkında tab'ı"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # About Group
        group_about = QGroupBox("ℹ️ Hakkında")
        group_about.setStyleSheet(self.get_group_style())
        about_layout = QVBoxLayout(group_about)
        
        # App info
        app_name = QLabel("Fizyonomi Analiz Asistanı")
        app_name.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        about_layout.addWidget(app_name)
        
        version = QLabel("Versiyon: 1.0.0")
        about_layout.addWidget(version)
        
        author = QLabel("© 2024 Yasin Kılınç")
        about_layout.addWidget(author)
        
        about_layout.addWidget(QLabel(""))
        
        # Description
        desc = QLabel(
            "3D yüz rekonstrüksiyonu ve analizi için gelişmiş araç.\n"
            "MediaPipe ve PyQt6 kullanılarak geliştirilmiştir."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a6adc8;")
        about_layout.addWidget(desc)
        
        layout.addWidget(group_about)
        
        # System Info Group
        group_system = QGroupBox("💻 Sistem Bilgisi")
        group_system.setStyleSheet(self.get_group_style())
        system_layout = QVBoxLayout(group_system)
        
        import platform
        import sys
        
        system_layout.addWidget(QLabel(f"Python: {sys.version.split()[0]}"))
        system_layout.addWidget(QLabel(f"İşletim Sistemi: {platform.system()} {platform.release()}"))
        system_layout.addWidget(QLabel(f"Mimari: {platform.machine()}"))
        
        layout.addWidget(group_system)
        
        layout.addStretch()
        return tab
    
    def get_group_style(self):
        """Grup kutusu stili"""
        return """
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: #181825;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #89b4fa;
            }
        """

    def save_settings(self):
        """Ayarları kaydet ve uygula"""
        # Collect all settings
        settings = {
            'theme': self.combo_theme.currentIndex(),
            'auto_enhance': self.chk_auto_enhance.isChecked(),
            'auto_brightness': self.chk_auto_brightness.isChecked(),
            'noise_reduction': self.slider_noise.value(),
            'camera_resolution': self.combo_camera_res.currentIndex(),
            'mesh_distance': self.spin_mesh_distance.value(),
            'mesh_elevation': self.spin_mesh_elevation.value(),
            'mesh_azimuth': self.spin_mesh_azimuth.value(),
            'smoothing': self.chk_smoothing.isChecked(),
            'detail_level': self.combo_detail.currentIndex(),
            'color_correct': self.chk_color_correct.isChecked(),
            'precision': self.chk_precision.isChecked(),
            'landmarks': self.combo_landmarks.currentIndex(),
            'confidence': self.spin_confidence.value(),
            'multi_face': self.chk_multi_face.isChecked(),
            'quality_threshold': self.spin_quality_threshold.value(),
            'auto_retry': self.chk_auto_retry.isChecked(),
            'font_size': self.combo_font_size.currentIndex(),
            'font_size': self.combo_font_size.currentIndex(),
            'show_grid': self.chk_show_grid.isChecked(),
            'archive_refresh_interval': self.spin_refresh_interval.value()
        }
        
        # Save to database
        from desktop_app.database import Database
        db = Database()
        db.save_settings_batch(settings)
        
        # Apply theme
        theme_idx = self.combo_theme.currentIndex()
        app = QApplication.instance()
        if theme_idx == 0:  # Dark
            app.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e2e;
                    color: #cdd6f4;
                }
                QLabel { color: #cdd6f4; }
                QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
                QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
                QGroupBox { color: #cdd6f4; border: 1px solid #45475a; }
                QCheckBox { color: #cdd6f4; }
                QSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }
            """)
        elif theme_idx == 1:  # Light
            app.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #eff1f5;
                    color: #4c4f69;
                }
                QLabel { color: #4c4f69; }
                QPushButton { background-color: #ccd0da; color: #4c4f69; border: 1px solid #bcc0cc; }
                QComboBox { background-color: #e6e9ef; color: #4c4f69; border: 1px solid #bcc0cc; }
                QGroupBox { color: #4c4f69; border: 1px solid #bcc0cc; }
                QCheckBox { color: #4c4f69; }
                QSpinBox { background-color: #e6e9ef; color: #4c4f69; border: 1px solid #bcc0cc; }
            """)
        else:  # System
            app.setStyleSheet("")
        
        # Emit signal
        self.settings_saved.emit("✅ Ayarlar kaydedildi ve uygulandı")
    
    def load_settings(self):
        """Ayarları database'den yükle"""
        from desktop_app.database import Database
        db = Database()
        settings = db.get_all_settings()
        
        # Apply to UI with defaults
        self.combo_theme.setCurrentIndex(int(settings.get('theme', 0)))
        self.chk_auto_enhance.setChecked(settings.get('auto_enhance', 'True') == 'True')
        self.chk_auto_brightness.setChecked(settings.get('auto_brightness', 'False') == 'True')
        self.slider_noise.setValue(int(settings.get('noise_reduction', 5)))
        self.combo_camera_res.setCurrentIndex(int(settings.get('camera_resolution', 1)))
        self.spin_mesh_distance.setValue(int(settings.get('mesh_distance', 400)))
        self.spin_mesh_elevation.setValue(int(settings.get('mesh_elevation', 70)))
        self.spin_mesh_azimuth.setValue(int(settings.get('mesh_azimuth', 0)))
        self.chk_smoothing.setChecked(settings.get('smoothing', 'True') == 'True')
        self.combo_detail.setCurrentIndex(int(settings.get('detail_level', 1)))
        self.chk_color_correct.setChecked(settings.get('color_correct', 'False') == 'True')
        self.chk_precision.setChecked(settings.get('precision', 'False') == 'True')
        self.combo_landmarks.setCurrentIndex(int(settings.get('landmarks', 1)))
        self.spin_confidence.setValue(int(settings.get('confidence', 50)))
        self.chk_multi_face.setChecked(settings.get('multi_face', 'False') == 'True')
        self.spin_quality_threshold.setValue(int(settings.get('quality_threshold', 60)))
        self.chk_auto_retry.setChecked(settings.get('auto_retry', 'False') == 'True')
        self.combo_font_size.setCurrentIndex(int(settings.get('font_size', 1)))
        self.chk_show_grid.setChecked(settings.get('show_grid', 'True') == 'True')
        self.spin_refresh_interval.setValue(int(settings.get('archive_refresh_interval', 0)))
