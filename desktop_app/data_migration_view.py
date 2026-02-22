from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTextEdit, QLabel, QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from annotation_engine import AnnotationManager

class MigrationWorker(QThread):
    """Background thread for migration"""
    progress = Signal(str) # Log message
    finished = Signal(int) # Total count
    
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        
    def run(self):
        self.progress.emit("Göç işlemi başlatılıyor...")
        try:
            # We need to modify migrate_from_files to yield progress or we just wrap it
            # For now, we'll capture stdout or modify the manager later.
            # Let's just run it and report start/end since it's synchronous currently
            count = self.manager.migrate_from_files()
            self.progress.emit(f"İşlem tamamlandı. Toplam {count} resim aktarıldı.")
            self.finished.emit(count)
        except Exception as e:
            self.progress.emit(f"Hata oluştu: {str(e)}")
            self.finished.emit(0)

class DataMigrationView(QWidget):
    """View for managing data migration to database"""
    
    def __init__(self):
        super().__init__()
        self.manager = AnnotationManager()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("📂 Veri Tabanı Aktarımı")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(header)
        
        desc = QLabel("Dosya sistemindeki (dataset/processed ve dataset/annotations) verileri veritabanına aktarır.\nZaten veritabanında olan veriler atlanır.")
        desc.setStyleSheet("color: #a6adc8;")
        layout.addWidget(desc)
        
        # Controls
        btn_layout = QHBoxLayout()
        
        self.btn_migrate = QPushButton("🚀 Aktarımı Başlat")
        self.btn_migrate.setMinimumHeight(50)
        self.btn_migrate.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 14px;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        """)
        self.btn_migrate.clicked.connect(self.start_migration)
        btn_layout.addWidget(self.btn_migrate)
        
        layout.addLayout(btn_layout)
        
        # Log Output
        layout.addWidget(QLabel("İşlem Günlüğü:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #181825;
                color: #a6adc8;
                font-family: 'Consolas', 'Monaco', monospace;
                border: 1px solid #313244;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.log_output)
        
    def log(self, message):
        self.log_output.append(message)
        # Auto scroll
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def start_migration(self):
        self.btn_migrate.setEnabled(False)
        self.log("--- Aktarım Başlatıldı ---")
        
        self.worker = MigrationWorker(self.manager)
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        
    def on_finished(self, count):
        self.btn_migrate.setEnabled(True)
        self.log(f"--- İşlem Bitti. Aktarılan: {count} ---")
