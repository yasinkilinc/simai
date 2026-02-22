#!/usr/bin/env python3
"""
Startup Test Script
Tests if the application can start without errors
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        print("  ├─ Importing desktop_app.main_window...")
        from desktop_app.main_window import MainWindow
        print("  ├─ Importing desktop_app.camera_view...")
        from desktop_app.camera_view import CameraView
        print("  ├─ Importing desktop_app.analysis_view...")
        from desktop_app.analysis_view import AnalysisView
        print("  ├─ Importing desktop_app.archive_view...")
        from desktop_app.archive_view import ArchiveView
        print("  ├─ Importing desktop_app.settings_view...")
        from desktop_app.settings_view import SettingsView
        print("  ├─ Importing desktop_app.database...")
        from desktop_app.database import Database
        print("  └─ All imports successful!")
        return True
    except ImportError as e:
        print(f"  └─ ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  └─ ❌ Unexpected error: {e}")
        return False


def test_app_creation():
    """Test that QApplication can be created"""
    print("\n🔍 Testing QApplication creation...")
    
    try:
        from PySide6.QtWidgets import QApplication
        
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            print("  ├─ Creating new QApplication...")
            app = QApplication(sys.argv)
        else:
            print("  ├─ Using existing QApplication...")
        
        print("  └─ QApplication created successfully!")
        return True, app
    except Exception as e:
        print(f"  └─ ❌ Failed to create QApplication: {e}")
        return False, None


def test_window_creation(app):
    """Test that MainWindow can be instantiated"""
    print("\n🔍 Testing MainWindow creation...")
    
    try:
        from desktop_app.main_window import MainWindow
        
        print("  ├─ Instantiating MainWindow...")
        window = MainWindow()
        print("  ├─ Window created successfully!")
        
        # Don't show the window in test mode
        # window.show()
        
        print("  └─ MainWindow instantiation successful!")
        return True
    except Exception as e:
        print(f"  └─ ❌ Failed to create MainWindow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all startup tests"""
    print("="*60)
    print("🚀 Face Personality App - Startup Test")
    print("="*60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed!")
        sys.exit(1)
    
    # Test 2: QApplication
    success, app = test_app_creation()
    if not success:
        print("\n❌ QApplication test failed!")
        sys.exit(1)
    
    # Test 3: MainWindow
    if not test_window_creation(app):
        print("\n❌ Window creation test failed!")
        sys.exit(1)
    
    # All tests passed
    print("\n" + "="*60)
    print("✅ All startup tests passed!")
    print("="*60)
    sys.exit(0)


if __name__ == '__main__':
    main()
