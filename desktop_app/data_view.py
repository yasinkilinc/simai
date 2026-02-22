from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from desktop_app.data_collection_view import DataCollectionView
from desktop_app.data_preprocessing_view import DataPreprocessingView

class DataView(QWidget):
    """Unified Data view with tabs for Collection and Preprocessing"""
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #45475a;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background-color: #313244;
                color: #cdd6f4;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background-color: #45475a;
            }
        """)
        
        # Tab 1: Data Collection
        self.data_collection_view = DataCollectionView()
        self.tabs.addTab(self.data_collection_view, "📥 Veri Seti Toplama")
        
        # Tab 2: Data Preprocessing
        self.data_preprocessing_view = DataPreprocessingView()
        self.tabs.addTab(self.data_preprocessing_view, "🔧 Veri Standartları")
        
        # Tab 3: Data Migration
        from desktop_app.data_migration_view import DataMigrationView
        self.data_migration_view = DataMigrationView()
        self.tabs.addTab(self.data_migration_view, "📂 Veri Aktarımı")
        
        
        layout.addWidget(self.tabs)
