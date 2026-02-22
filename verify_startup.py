import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔍 Verifying application startup...")

try:
    print("   Importing main modules...")
    from desktop_app.main_window import MainWindow
    from desktop_app.analysis_view import AnalysisView
    from desktop_app.data_collection_view import DataCollectionView
    print("   ✅ Imports successful.")
    
    # Optional: Try to instantiate classes if possible without GUI
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # print("   ✅ MainWindow initialized.")
    
    print("🎉 Startup verification passed!")
    sys.exit(0)
    
except Exception as e:
    print("\n❌ STARTUP VERIFICATION FAILED!")
    print(f"Error: {str(e)}")
    traceback.print_exc()
    sys.exit(1)
