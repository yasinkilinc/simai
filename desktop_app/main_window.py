"""
Ana pencere - Navigasyon ve content yönetimi
Modern, Material Design inspired dark theme
"""

import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QLabel, QStatusBar,
    QApplication
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QPixmap, QImage

# Import views
from desktop_app.camera_view import CameraView
from desktop_app.analysis_view import AnalysisView
from desktop_app.settings_view import SettingsView
# from desktop_app.archive_view import ArchiveView
# from desktop_app.settings_view import SettingsView
from desktop_app.ai_view import AIView


class MainWindow(QMainWindow):
    """Ana uygulama penceresi"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Fizyonomi AI - 3D Yüz Analizi")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Setup UI
        self.setup_ui()
        
        # Restore window geometry
        # self.restore_settings()
    
    def setup_ui(self):
        """UI bileşenlerini oluştur"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Left navigation panel
        nav_panel = self.create_nav_panel()
        main_layout.addWidget(nav_panel)
        
        # Content area (stacked widget for different views)
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, stretch=1)
        
        # Create views
        self.camera_view = CameraView()
        self.camera_view.photo_ready.connect(self.on_photo_ready)
        self.camera_view.status_message.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.content_stack.addWidget(self.camera_view)
        
        # Analysis view (real one now!)
        self.analysis_view = AnalysisView()
        self.analysis_view.status_message.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.content_stack.addWidget(self.analysis_view)
        
        # Page 2: Archive View
        from desktop_app.archive_view import ArchiveView
        self.archive_view = ArchiveView()
        self.archive_view.load_analysis.connect(self.load_analysis_from_archive)
        self.archive_view.status_message.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.content_stack.addWidget(self.archive_view)
        
        # Connect Analysis Save to Archive Refresh
        self.analysis_view.analysis_saved.connect(self.archive_view.refresh_archive)
        
        # Page 3: Settings View
        self.settings_view = SettingsView()
        self.settings_view.settings_saved.connect(lambda msg: self.statusBar().showMessage(msg, 3000))
        # Timer removed - archive now uses DB BLOBs, no auto-refresh needed
        self.content_stack.addWidget(self.settings_view)
        
        # Page 4: Unified Data View (Collection + Preprocessing in tabs)
        from desktop_app.data_view import DataView
        self.data_view = DataView()
        self.content_stack.addWidget(self.data_view)
        
        # Page 5: Annotation View
        from desktop_app.annotation_view import AnnotationView
        self.annotation_view = AnnotationView()
        self.content_stack.addWidget(self.annotation_view)

        # Page 6: AI Training View
        self.ai_view = AIView()
        self.content_stack.addWidget(self.ai_view)
        
        # Load settings from database on startup
        self.settings_view.load_settings()
        
        # Apply settings to camera view (resolution, mesh orientation)
        from desktop_app.database import Database
        db = Database()
        settings = db.get_all_settings()
        self.camera_view.apply_settings(settings)
        
        # Apply loaded settings (theme etc.)
        self.settings_view.save_settings()
        
        # Status bar
        self.statusBar().showMessage("Hazır", 3000)
    
    def create_nav_panel(self):
        """Sol navigasyon panelini oluştur"""
        panel = QWidget()
        panel.setObjectName("navPanel")
        panel.setFixedWidth(250)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(10)
        
        # App title/logo
        title_label = QLabel("🧠 Fizyonomi AI")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        subtitle = QLabel("3D Yüz Analizi")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Navigation buttons
        self.nav_buttons = []
        
        btn_camera = self.create_nav_button("📸 Kamera", 0)
        layout.addWidget(btn_camera)
        
        btn_analysis = self.create_nav_button("📊 Analiz", 1)
        layout.addWidget(btn_analysis)
        
        btn_archive = self.create_nav_button("📚 Arşiv", 2)
        layout.addWidget(btn_archive)
        
        btn_settings = self.create_nav_button("⚙️ Ayarlar", 3)
        layout.addWidget(btn_settings)
        
        btn_data = self.create_nav_button("📊 Veri", 4)
        layout.addWidget(btn_data)
        
        btn_annotation = self.create_nav_button("🏷️ Etiketleme", 5)
        layout.addWidget(btn_annotation)

        btn_ai = self.create_nav_button("🤖 AI Eğitim", 6)
        layout.addWidget(btn_ai)
        
        layout.addStretch()
        
        # ML status indicator
        ml_status = QLabel("🤖 ML: Hazır")
        ml_status.setObjectName("mlStatus")
        layout.addWidget(ml_status)
        
        # Version info
        version = QLabel("v1.0.0")
        version.setAlignment(Qt.AlignCenter)
        version.setObjectName("version")
        layout.addWidget(version)
        
        return panel
    
    def create_nav_button(self, text, index):
        """Navigasyon butonu oluştur"""
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setObjectName("navButton")
        btn.setMinimumHeight(50)
        btn.clicked.connect(lambda: self.switch_view(index))
        
        # İlk buton varsayılan aktif
        if index == 0:
            btn.setChecked(True)
        
        self.nav_buttons.append(btn)
        return btn
    
    def switch_view(self, index):
        """Görünümü değiştir"""
        self.content_stack.setCurrentIndex(index)
        
        # Update button states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        
        # Update status bar
        views = ["Kamera", "Analiz", "Arşiv", "Ayarlar", "Veri", "Etiketleme", "AI Eğitim"]
        self.statusBar().showMessage(f"{views[index]} görünümü", 3000)
    
    def apply_dark_theme(self):
        """Karanlık tema uygula"""
        # Basic dark theme stylesheet
        # qt-material kullanırsak daha iyi olur ama şimdilik basit
        stylesheet = """
            QMainWindow {
                background-color: #1e1e2e;
            }
            
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
                font-size: 13px;
            }
            
            #navPanel {
                background-color: #181825;
                border-right: 1px solid #313244;
            }
            
            #navButton {
                background-color: transparent;
                border: none;
                border-radius: 8px;
                padding: 12px;
                text-align: left;
                font-size: 14px;
                color: #cdd6f4;
            }
            
            #navButton:hover {
                background-color: #313244;
            }
            
            #navButton:checked {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            
            #subtitle, #version {
                color: #9399b2;
                font-size: 11px;
            }
            
            #mlStatus {
                background-color: #313244;
                padding: 8px;
                border-radius: 6px;
                font-size: 12px;
            }
            
            QLabel {
                color: #cdd6f4;
            }
            
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #b4befe;
            }
            
            QPushButton:pressed {
                background-color: #74c7ec;
            }
            
            QPushButton:disabled {
                background-color: #45475a;
                color: #6c7086;
            }
            
            QStatusBar {
                background-color: #181825;
                color: #9399b2;
                border-top: 1px solid #313244;
            }
        """
        
        self.setStyleSheet(stylesheet)
    
    def on_photo_ready(self, images_data):
        """Fotoğraf analiz için hazır - gerçek analizi başlat"""
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import QThread
        from desktop_app.logger import get_logger
        import sys
        import os
        from datetime import datetime
        
        # Extract images
        if isinstance(images_data, dict):
            image = images_data['front']
            side_image = images_data.get('side')
        else:
            image = images_data
            side_image = None
            

        logger = get_logger()
        logger.info("=" * 50)
        logger.info("Analiz başlatılıyor...")

        # Add src to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Show progress dialog
        progress = QProgressDialog("Analiz ediliyor...", None, 0, 5, self)
        progress.setWindowTitle("Fizyonomi Analizi")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        try:
            # 1. Yüz Hizalama (Data Standards)
            logger.info("Step 1/5: Yüz hizalama başlıyor...")
            progress.setLabelText("1/5: Yüz hizalanıyor...")
            progress.setValue(1)
            QApplication.processEvents()

            from src.preprocessing import FacePreprocessor
            preprocessor = FacePreprocessor()
            
            front_img = images_data['front']
            side_img = images_data['side']
            
            # Align front image
            aligned_front = preprocessor.align_face(front_img)
            if aligned_front is None:
                logger.warning("Yüz hizalanamadı!")
                progress.close()
                self.statusBar().showMessage("⚠️ Yüz tespit edilemedi veya hizalanamadı! Lütfen yüzünüzü net gösterin.", 5000)
                return
            images_data['front'] = aligned_front # Update data
            logger.info(f"✓ Yüz hizalandı. Original: {front_img.shape}, Aligned: {aligned_front.shape}")
            
            # DEBUG: Check if aligned image is actually different
            if np.array_equal(front_img, aligned_front):
                logger.warning("⚠️ Aligned image is IDENTICAL to original! Alignment might have failed silently.")
            else:
                logger.info("✓ Aligned image is different from original.")


            # 2. Otomatik Etiketleme (Auto Annotation)
            logger.info("Step 2/5: Otomatik etiketleme başlıyor...")
            progress.setLabelText("2/5: Otomatik etiketleme yapılıyor...")
            progress.setValue(2)
            QApplication.processEvents()

            from annotation_engine import AutoAnnotator
            annotator = AutoAnnotator()
            
            # Ensure contiguous (redundant but safe)
            if not aligned_front.flags['C_CONTIGUOUS']:
                logger.warning("Aligned front is NOT contiguous! Fixing...")
                aligned_front = np.ascontiguousarray(aligned_front)
            
            # Debug: Save the exact image used for detection
            # DEBUG: Save for inspection - DISABLED
            # cv2.imwrite("debug_aligned_input.jpg", aligned_front)
            # logger.info(f"DEBUG: Saved debug_aligned_input.jpg. Shape: {aligned_front.shape}, Contiguous: {aligned_front.flags['C_CONTIGUOUS']}")

            # Get landmarks from aligned image
            landmarks = annotator.get_landmarks(aligned_front)
            
            if landmarks:
                lm0 = landmarks[0]
                logger.info(f"DEBUG: Detected Landmarks[0]: x={lm0.x:.4f}, y={lm0.y:.4f}")
            
            if not landmarks:
                logger.warning("Yüz tespit edilemedi!")
                progress.close()
                self.statusBar().showMessage("⚠️ Yüz tespit edilemedi! Lütfen yüzünüzü net gösterin ve iyi aydınlatma sağlayın.", 5000)
                return
            # Step 2.5: Side profile reconstruction (if available)
            side_landmarks = None
            
            # Generate annotations
            annotations = annotator.annotate_image(aligned_front)
            logger.info("✓ Otomatik etiketleme tamamlandı")

            # 3. 3D Mesh ve Özellik Çıkarımı
            logger.info("Step 3/5: 3D model ve özellik çıkarımı...")
            progress.setLabelText("3/5: 3D analiz yapılıyor...")
            progress.setValue(3)
            QApplication.processEvents()

            from src.reconstruction import FaceReconstructor
            from src.features import FaceFeatures
            from src.interpreter import PhysiognomyInterpreter
            from src.visualization import Visualizer
            
            reconstructor = FaceReconstructor()
            interpreter = PhysiognomyInterpreter()
            
            # 3D Reconstruction
            # Note: landmarks already obtained from AutoAnnotator (which uses MediaPipe)
            # But FaceReconstructor also uses MediaPipe. 
            # We can reuse landmarks if they are compatible, or just re-run process_frame to be safe 
            # and get the specific format FaceReconstructor expects if different.
            # AutoAnnotator returns NormalizedLandmarkList. FaceReconstructor.process_frame returns the same.
            # So we can just use get_3d_points.
            
            points_3d = reconstructor.get_3d_points(landmarks, aligned_front.shape)
            mesh_landmarks = landmarks # Reuse landmarks
            
            # Feature Extraction
            # FaceFeatures(landmarks_3d, frame=None, side_landmarks=None, side_frame=None)
            features = FaceFeatures(points_3d, frame=aligned_front, annotations=annotations)
            logger.info("✓ Özellikler çıkarıldı")
            
            # 4. Yorumlama (Annotations ile)
            logger.info("Step 4/5: Yorumlama...")
            progress.setLabelText("4/5: Kişilik analizi yapılıyor...")
            progress.setValue(4)
            QApplication.processEvents()
            
            report = interpreter.interpret(features)
            logger.info("✓ Rapor oluşturuldu")
            
            # 5. Görselleştirme (Visualization)
            logger.info("Step 5/5: Görselleştirme...")
            progress.setLabelText("5/5: Sonuçlar hazırlanıyor...")
            progress.setValue(5)
            QApplication.processEvents()
            
            # Draw measurements on a copy of aligned image
            # Convert to QPixmap for drawing
            vis_image = QPixmap.fromImage(QImage(aligned_front.data, aligned_front.shape[1], aligned_front.shape[0], aligned_front.strides[0], QImage.Format_RGB888).rgbSwapped())
            vis_image = Visualizer.draw_measurements(vis_image, mesh_landmarks, annotator)
            
            # 6. Sonuçları Göster
            self.analysis_view.display_results(
                report, 
                vis_image, # Visualized image (QPixmap)
                points_3d, 
                mesh_landmarks, 
                side_image=side_img,
                clean_image=aligned_front # Pass clean image (numpy) for saving
            )
            
            self.switch_view(1) # Analiz sekmesine git
            self.statusBar().showMessage("✅ Analiz tamamlandı.", 5000)
            progress.close()
            
        except Exception as e:
            logger.error(f"Analiz hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            progress.close()
            self.statusBar().showMessage(f"❌ Hata: {str(e)}", 5000)
            
        except Exception as e:
            logger.error(f"Analiz hatası: {str(e)}", exc_info=True)
            progress.close()
            self.statusBar().showMessage(f"❌ Analiz hatası: {str(e)}", 10000)
            import traceback
            traceback.print_exc()
    
    def show_success_banner(self, report):
        """Başarılı analiz banner'ı göster (popup yerine)"""
        from PySide6.QtWidgets import QGraphicsOpacityEffect
        from PySide6.QtCore import QTimer, QPropertyAnimation
        
        # Banner widget oluştur
        banner = QLabel(self)
        banner.setText(
            f"✅ Analiz Tamamlandı! | "
            f"Yüz: {report['face_shape']} | "
            f"Olumlu: {len(report['analysis']['positive'])} | "
            f"Dikkat: {len(report['analysis']['negative'])}"
        )
        banner.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #a6e3a1, stop:1 #89b4fa);
            color: #1e1e2e;
            font-size: 14px;
            font-weight: bold;
            padding: 15px;
            border-radius: 0px;
        """)
        banner.setAlignment(Qt.AlignCenter)
        banner.setFixedHeight(50)
        banner.setGeometry(0, 0, self.width(), 50)
        banner.show()
        
        # 3 saniye sonra fade out
        QTimer.singleShot(3000, lambda: banner.deleteLater())
        
    def load_analysis_from_archive(self, row):
        """Arşivden analiz yükle ve göster"""
        self.switch_view(1) # Switch to Analysis View
        self.analysis_view.load_analysis_from_history(row)
        self.statusBar().showMessage(f"📂 Arşivden yüklendi: {row['timestamp']}", 3000)
        
    def closeEvent(self, event):
        """Uygulama kapatılırken"""
        if hasattr(self, 'camera_view'):
            self.camera_view.cleanup()
        if hasattr(self, 'ai_view'):
            self.ai_view.cleanup()
        event.accept()
