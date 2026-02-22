# -*- coding: utf-8 -*-
"""
Ana uygulama giriş noktası
Splash screen ve ana pencereyi başlatır
"""

import sys
import os
from datetime import datetime
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QTimer

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
import logging
from desktop_app.logger import get_logger

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging globally
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# logger = get_logger() # Moved inside main for try-except scope

from desktop_app.main_window import MainWindow


def main():
    """Ana uygulama giriş noktası"""
    try:
        logger = get_logger()
        logger.info("=" * 50)
        logger.info("Fizyonomi AI Desktop Uygulaması Başlatıldı")
        logger.info(f"Log dosyası: {os.path.abspath('logs/app.log')}")
        logger.info("=" * 50)
        
        # Create QApplication
        logger.info("QApplication oluşturuluyor...")
        app = QApplication(sys.argv)
        app.setApplicationName("Fizyonomi AI")
        app.setOrganizationName("Fizyonomi")
        app.setApplicationVersion("1.0.0")
        
        # Create main window
        logger.info("Ana pencere oluşturuluyor...")
        main_window = MainWindow()
        main_window.show()
        
        # Show success
        logger.info("Uygulama gösteriliyor...")
        logger.info("Uygulama başlatıldı - Kullanıcı arayüzü hazır")
        
        # Start event loop
        sys.exit(app.exec())
        
    except ImportError as e:
        error_msg = f"❌ Import hatası: {str(e)}\n\nLütfen gerekli bağımlılıkları yükleyin:\npip install -r requirements.txt"
        print(error_msg)
        if 'logger' in locals():
            logger.error(error_msg, exc_info=True)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"❌ Beklenmeyen hata: {str(e)}\n\nDetaylar için logs/app.log dosyasını kontrol edin."
        print(error_msg)
        
        # Try to log if logger is available
        if 'logger' in locals():
            logger.error("Uygulama başlatılamadı", exc_info=True)
            
            # Create crash report
            import traceback
            crash_file = f"logs/crash-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(crash_file), exist_ok=True)
            with open(crash_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("FACE PERSONALITY APP - CRASH REPORT\n")
                f.write("="*60 + "\n\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            print(f"Crash report kaydedildi: {crash_file}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
