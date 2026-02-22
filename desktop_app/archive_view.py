from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QPushButton, QScrollArea, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from desktop_app.database import Database

class ArchiveView(QWidget):
    """Tam ekran arşiv görünümü - DB BLOB tabanlı"""
    
    # Signal to load an analysis
    load_analysis = Signal(dict) # row data
    status_message = Signal(str) # status bar message
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("🗂 Analiz Arşivi")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #cdd6f4;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Removed Refresh Button as requested
        
        layout.addLayout(header_layout)
        
        # Grid Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background-color: transparent; }
            QWidget { background-color: transparent; }
        """)
        
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        scroll.setWidget(self.grid_container)
        layout.addWidget(scroll)
        
        # Initial load
        self.refresh_archive()
        
    def refresh_archive(self):
        """Arşivi veritabanından yükle (BLOB'lardan)"""
        print("DEBUG: ArchiveView.refresh_archive called")
        # Clear grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        try:
            db = Database()
            analyses = db.get_recent_analyses(limit=100)
            
            print(f"DEBUG ArchiveView: {len(analyses)} kayıt bulundu")
            
            row, col = 0, 0
            max_cols = 4
            
            for analysis in analyses:
                card = self.create_analysis_card(analysis, db)
                self.grid_layout.addWidget(card, row, col)
                
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
                    
            if not analyses:
                empty_label = QLabel("Henüz kaydedilmiş analiz yok.")
                empty_label.setStyleSheet("color: #6c7086; font-size: 16px; font-style: italic;")
                self.grid_layout.addWidget(empty_label, 0, 0)
                
        except Exception as e:
            print(f"Arşiv yükleme hatası: {e}")
            import traceback
            traceback.print_exc()
            
    def create_analysis_card(self, row, db):
        """Tek bir analiz kartı oluştur - BLOB'dan"""
        card = QFrame()
        card.setFixedSize(250, 300)
        card.setStyleSheet("""
            QFrame {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 12px;
            }
            QFrame:hover {
                border: 1px solid #89b4fa;
                background-color: #45475a;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Image from BLOB
        img_label = QLabel()
        img_label.setFixedSize(230, 180)
        img_label.setStyleSheet("background-color: #1e1e2e; border-radius: 8px; border: none;")
        img_label.setAlignment(Qt.AlignCenter)
        
        photo_id = row['id']
        timestamp = row.get('timestamp', 'Bilinmiyor')
        
        # Load thumbnail from BLOB
        try:
            import sqlite3
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT front_image FROM photos WHERE id = ?", (photo_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                # Decode BLOB
                nparr = np.frombuffer(result[0], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Convert to QPixmap
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    scaled = pixmap.scaled(img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    img_label.setPixmap(scaled)
                else:
                    img_label.setText("Hata")
            else:
                img_label.setText("Resim\nYok")
        except Exception as e:
            print(f"Thumbnail error for ID {photo_id}: {e}")
            img_label.setText("Hata")
            
        layout.addWidget(img_label)
        
        # Info
        info_layout = QVBoxLayout()
        
        # Show Date from timestamp
        lbl_date = QLabel(f"📅 {timestamp}")
        lbl_date.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold; border: none; background: transparent;")
        lbl_date.setWordWrap(True)
        info_layout.addWidget(lbl_date)
        
        # Show Face Shape
        face_shape = row.get('face_shape', 'Bilinmiyor')
        lbl_shape = QLabel(f"👤 {face_shape}")
        lbl_shape.setStyleSheet("color: #a6adc8; font-size: 11px; border: none; background: transparent;")
        info_layout.addWidget(lbl_shape)
        
        layout.addLayout(info_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        btn_load = QPushButton("Aç")
        btn_load.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        btn_load.clicked.connect(lambda: self.load_analysis.emit(row))
        btn_layout.addWidget(btn_load)
        
        # Delete button
        btn_del = QPushButton("Sil")
        btn_del.setStyleSheet("""
            QPushButton {
                background-color: #f38ba8;
                color: #1e1e2e;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #eba0ac; }
        """)
        btn_del.clicked.connect(lambda: self.delete_analysis(row))
        btn_layout.addWidget(btn_del)
        
        layout.addLayout(btn_layout)
        
        return card

    def delete_analysis(self, row):
        """Analizi sil (Sadece DB - BLOB zaten içinde)"""
        reply = QMessageBox.question(
            self, "Silme Onayı", 
            "Bu analizi silmek istediğinize emin misiniz?\nBu işlem geri alınamaz.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                db = Database()
                db.delete_analysis(row['id'])
                
                # Refresh UI
                self.refresh_archive()
                
                # Emit status message instead of popup
                self.status_message.emit("✅ Analiz silindi.")
                
            except Exception as e:
                self.status_message.emit(f"❌ Silme hatası: {str(e)}")
