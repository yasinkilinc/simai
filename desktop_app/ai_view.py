from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QFileDialog, QTextEdit,
    QGroupBox, QProgressBar, QComboBox, QMessageBox, 
    QRadioButton, QButtonGroup, QTabWidget, QFormLayout,
    QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QFont, QIcon

import os
import sys
import time
from datetime import timedelta

class TrainingThread(QThread):
    """Background thread for training"""
    log_signal = Signal(str)
    progress_signal = Signal(float)
    finished_signal = Signal(bool, str)
    
    def __init__(self, data_path, model_path, target_column, model_type='rf', backbone='resnet50', epochs=25, batch_size=16, accumulation_steps=1):
        super().__init__()
        self.data_path = data_path
        self.model_path = model_path
        self.target_column = target_column
        self.model_type = model_type
        self.backbone = backbone
        self.epochs = epochs
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        
    def run(self):
        self.start_time = time.time()
        try:
            self.log_signal.emit(f"Eğitim başlatılıyor...\nTip: {self.model_type.upper()}\nVeri: {self.data_path}\nHedef: {self.target_column}")
            
            if self.model_type == 'rf':
                # Random Forest Training
                from src.ml_engine import MLEngine
                engine = MLEngine()
                self.log_signal.emit("Veriler yükleniyor (Random Forest)...")
                success = engine.train(self.data_path, target_column=self.target_column)
                if success:
                    engine.save_model(self.model_path)
                    self.finished_signal.emit(True, "RF Modeli başarıyla eğitildi!")
                else:
                    self.finished_signal.emit(False, "RF Eğitimi başarısız oldu.")
                    
            elif self.model_type == 'dl':
                # Deep Learning Training
                self.log_signal.emit(f"Derin Öğrenme Başlatılıyor (Backbone: {self.backbone})...")
                self.log_signal.emit("NOT: Bu işlem GPU/CPU hızına göre uzun sürebilir.")
                
                import subprocess
                
                # Construct command
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.join(base_dir, "scripts", "train_deep_model.py")
                img_dir = os.path.join(base_dir, "dataset", "export", "images")
                
                cmd = [
                    sys.executable, script_path,
                    '--data_csv', self.data_path,
                    '--img_dir', img_dir,
                    '--batch_size', str(self.batch_size),
                    '--epochs', str(self.epochs),
                    '--accumulation_steps', str(self.accumulation_steps),
                    '--output_path', self.model_path
                ]
                
                self.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                
                # Stream output
                for line in self.process.stdout:
                    line = line.strip()
                    if "[PROGRESS]" in line:
                        try:
                            # Parse [PROGRESS] current/total
                            parts = line.split("]")[1].strip().split("/")
                            current = float(parts[0])
                            total = float(parts[1])
                            percent = (current / total) * 100.0
                            self.progress_signal.emit(percent)
                        except:
                            pass
                    self.log_signal.emit(line)
                    
                self.process.wait()
                
                if self.process.returncode == 0:
                    self.finished_signal.emit(True, "Deep Learning Modeli başarıyla eğitildi!")
                else:
                    self.finished_signal.emit(False, "DL Eğitimi hata ile sonlandı.")

        except Exception as e:
            self.finished_signal.emit(False, f"Hata oluştu: {str(e)}")

    def stop(self):
        """Kill the subprocess if running"""
        if hasattr(self, 'process') and self.process:
            try:
                self.process.terminate()
                self.process.kill()
            except:
                pass

class PredictionThread(QThread):
    """Background thread for prediction to avoid UI freezing"""
    finished_signal = Signal(object) # Returns dict or None
    error_signal = Signal(str)
    
    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd
        
    def run(self):
        import subprocess
        import json
        try:
            # Run subprocess
            result = subprocess.run(self.cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout)
                    self.finished_signal.emit(output)
                except json.JSONDecodeError:
                    self.error_signal.emit(f"Çıktı okunamadı:\n{result.stdout}")
            else:
                self.error_signal.emit(f"Hata:\n{result.stderr}")
                
        except Exception as e:
            self.error_signal.emit(f"Çalıştırma Hatası: {str(e)}")

class AIView(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header = QLabel("🤖 Yapay Zeka Merkezi")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setStyleSheet("color: #cdd6f4; margin-bottom: 5px;")
        main_layout.addWidget(header)
        
        # Stylesheet
        self.setStyleSheet("""
            QWidget {
                color: #cdd6f4;
                font-family: 'Segoe UI', sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #313244;
                border-radius: 6px;
                background-color: #1e1e2e;
            }
            QTabBar::tab {
                background: #181825;
                color: #a6adc8;
                padding: 10px 20px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #313244;
                color: #cdd6f4;
                font-weight: bold;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 24px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                color: #89b4fa;
                font-weight: bold;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 6px;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #89b4fa;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                padding: 6px 12px;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #45475a;
            }
            QTextEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 4px;
                color: #a6adc8;
                font-family: monospace;
            }
            QProgressBar {
                border: 1px solid #313244;
                border-radius: 4px;
                text-align: center;
                background-color: #313244;
                color: #ffffff;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #2e7d32;
            }
        """)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Tab 1: Training
        self.tab_train = QWidget()
        self.setup_train_tab()
        self.tabs.addTab(self.tab_train, "🛠️ Model Eğitimi")
        
        # Tab 2: Prediction
        self.tab_predict = QWidget()
        self.setup_predict_tab()
        self.tabs.addTab(self.tab_predict, "🔮 Test ve Tahmin")
        
    def setup_train_tab(self):
        layout = QVBoxLayout(self.tab_train)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Configuration Group
        config_group = QGroupBox("Eğitim Ayarları")
        form_layout = QFormLayout(config_group)
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form_layout.setSpacing(15)
        
        # Model Type
        type_layout = QHBoxLayout()
        self.rb_rf = QRadioButton("Klasik (Random Forest)")
        self.rb_rf.setChecked(True)
        self.rb_dl = QRadioButton("Derin Öğrenme (Deep Learning)")
        self.bg_type = QButtonGroup()
        self.bg_type.addButton(self.rb_rf)
        self.bg_type.addButton(self.rb_dl)
        type_layout.addWidget(self.rb_rf)
        type_layout.addWidget(self.rb_dl)
        type_layout.addStretch()
        form_layout.addRow("Model Tipi:", type_layout)
        
        # DL Settings (Conditional)
        self.dl_widget = QWidget()
        dl_layout = QHBoxLayout(self.dl_widget)
        dl_layout.setContentsMargins(0,0,0,0)
        
        self.combo_backbone = QComboBox()
        self.combo_backbone.addItems(["efficientnet_b4", "resnet50"])
        self.combo_backbone.setFixedWidth(150)
        self.combo_backbone.setToolTip("Modelin görsel algı kapasitesi.\nEfficientNet-B4: Daha hassas ve modern (Önerilen)\nResNet50: Daha eski ama hızlı")
        
        self.spin_epochs = QLineEdit("25")
        self.spin_epochs.setFixedWidth(60)
        self.spin_epochs.setValidator(None) # Add int validator ideally
        self.spin_epochs.setToolTip("Eğitim tekrar sayısı.\nAz (5-10): Hızlı test\nOrta (25-50): İdeal öğrenme\nÇok (100+): Maksimum hassasiyet (Uzun sürer)")
        
        # GPU Cooling Control - Simple 3-position slider
        from PySide6.QtWidgets import QSlider
        
        self.slider_cooling = QSlider(Qt.Horizontal)
        self.slider_cooling.setRange(0, 2)  # 0=Auto, 1=Hızlı, 2=Yavaş
        self.slider_cooling.setValue(0)
        self.slider_cooling.setTickPosition(QSlider.TicksBelow)
        self.slider_cooling.setTickInterval(1)
        self.slider_cooling.setFixedWidth(150)
        
        self.label_cooling_mode = QLabel("Max Hız")
        self.label_cooling_mode.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: bold;")
        
        # Connect signal
        self.slider_cooling.valueChanged.connect(self.update_cooling_label)
        
        dl_layout.addWidget(QLabel("Backbone:"))
        dl_layout.addWidget(self.combo_backbone)
        dl_layout.addSpacing(20)
        dl_layout.addWidget(QLabel("Epochs:"))
        dl_layout.addWidget(self.spin_epochs)
        dl_layout.addSpacing(20)
        dl_layout.addWidget(QLabel("GPU Soğutma:"))
        dl_layout.addWidget(self.slider_cooling)
        dl_layout.addWidget(self.label_cooling_mode)
        dl_layout.addStretch()
        
        form_layout.addRow("DL Detayları:", self.dl_widget)
        self.dl_widget.hide() # Initially hidden
        
        self.bg_type.buttonClicked.connect(self.toggle_dl_settings)
        
        # Data Path
        data_layout = QHBoxLayout()
        self.data_path = QLineEdit()
        self.data_path.setText(os.path.join(os.getcwd(), "dataset", "export", "data.csv"))
        btn_browse_data = QPushButton("Seç")
        btn_browse_data.setFixedWidth(60)
        btn_browse_data.clicked.connect(self.browse_data)
        data_layout.addWidget(self.data_path)
        data_layout.addWidget(btn_browse_data)
        form_layout.addRow("Veri Seti (CSV):", data_layout)
        
        # Target
        self.combo_target = QComboBox()
        self.combo_target.addItems([
            "target_face_shape", 
            "target_forehead_width", "target_forehead_height",
            "target_eyes_size", "target_eyes_slant",
            "target_nose_length", "target_nose_width",
            "target_lips_upper", "target_lips_lower",
            "target_chin_prominence"
        ])
        self.combo_target.setEditable(True)
        form_layout.addRow("Hedef Özellik:", self.combo_target)
        
        # Output Path
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setText(os.path.join(os.getcwd(), "models", "face_personality_rf.joblib"))
        btn_browse_model = QPushButton("Seç")
        btn_browse_model.setFixedWidth(60)
        btn_browse_model.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(btn_browse_model)
        form_layout.addRow("Model Kayıt Yeri:", model_layout)
        
        layout.addWidget(config_group)
        
        # Action Button
        self.btn_train = QPushButton("🚀 Eğitimi Başlat")
        self.btn_train.setMinimumHeight(45)
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #94e2d5; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        """)
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)
        
        # Progress & Logs
        self.progress = QProgressBar()
        self.progress.setRange(0, 10000)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v") # Will be updated dynamically
        self.progress.hide()
        layout.addWidget(self.progress)
        
        log_group = QGroupBox("İşlem Günlüğü")
        log_layout = QVBoxLayout(log_group)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        log_layout.addWidget(self.log_area)
        layout.addWidget(log_group)
        
    def setup_predict_tab(self):
        layout = QVBoxLayout(self.tab_predict)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Input Group
        input_group = QGroupBox("Test Verisi")
        form_layout = QFormLayout(input_group)
        
        # Model Selection
        model_layout = QHBoxLayout()
        self.combo_test_model = QComboBox()
        self.combo_test_model.currentIndexChanged.connect(self.on_model_changed)
        btn_refresh = QPushButton("Yenile")
        btn_refresh.setFixedWidth(70)
        btn_refresh.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(self.combo_test_model)
        model_layout.addWidget(btn_refresh)
        form_layout.addRow("Model Seç:", model_layout)
        
        # Model Info
        self.lbl_model_info = QLabel("Model bilgisi yükleniyor...")
        self.lbl_model_info.setStyleSheet("color: #a6adc8; font-size: 11px; font-style: italic;")
        form_layout.addRow("", self.lbl_model_info)
        
        # Image Selection
        img_layout = QHBoxLayout()
        self.test_img_path = QLineEdit()
        self.test_img_path.setPlaceholderText("Analiz edilecek yüz fotoğrafını seçin...")
        btn_browse_img = QPushButton("Dosya Seç")
        btn_browse_img.clicked.connect(self.browse_test_image)
        img_layout.addWidget(self.test_img_path)
        img_layout.addWidget(btn_browse_img)
        form_layout.addRow("Fotoğraf:", img_layout)
        
        # Model Selection (Optional override)
        # Usually uses the one from training tab or a specific one
        
        layout.addWidget(input_group)
        
        # Predict Button
        self.btn_predict = QPushButton("🔮 Tahmin Et")
        self.btn_predict.setMinimumHeight(45)
        self.btn_predict.setStyleSheet("""
            QPushButton {
                background-color: #cba6f7;
                color: #1e1e2e;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #b4befe; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        """)
        self.btn_predict.clicked.connect(self.run_prediction)
        layout.addWidget(self.btn_predict)
        
        # Results
        res_group = QGroupBox("Analiz Sonuçları")
        res_layout = QVBoxLayout(res_group)
        self.result_area = QLabel("Henüz bir analiz yapılmadı.")
        self.result_area.setAlignment(Qt.AlignCenter)
        self.result_area.setStyleSheet("""
            QLabel {
                background-color: #181825;
                border-radius: 6px;
                padding: 20px;
                color: #a6adc8;
                font-size: 14px;
            }
        """)
        self.result_area.setWordWrap(True)
        res_layout.addWidget(self.result_area)
        layout.addWidget(res_group)
        
        layout.addStretch()
        
        # Initial load
        self.refresh_model_list()

    # --- Logic Methods ---
    
    def refresh_model_list(self):
        """models klasöründeki .pth dosyalarını listele"""
        self.combo_test_model.clear()
        models_dir = os.path.join(os.getcwd(), "models")
        if os.path.exists(models_dir):
            files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            self.combo_test_model.addItems(files)
            
        if self.combo_test_model.count() == 0:
            self.combo_test_model.addItem("Model bulunamadı")
            self.lbl_model_info.setText("Lütfen önce model eğitimi yapın.")
        else:
            self.on_model_changed()
            
    def on_model_changed(self):
        """Seçili model değiştiğinde metadata yükle"""
        model_name = self.combo_test_model.currentText()
        if not model_name or model_name == "Model bulunamadı":
            return
            
        models_dir = os.path.join(os.getcwd(), "models")
        json_path = os.path.join(models_dir, model_name.replace(".pth", ".json"))
        
        info_text = "Metadata bulunamadı."
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    date = meta.get('date', '-')
                    acc = meta.get('final_val_loss', '-')
                    backbone = meta.get('backbone', '-')
                    info_text = f"📅 {date} | 🧠 {backbone} | 📉 Loss: {acc}"
            except:
                pass
        
        self.lbl_model_info.setText(info_text)
    
    def update_cooling_label(self):
        """Update cooling mode label based on slider position"""
        mode = self.slider_cooling.value()
        if mode == 0:
            self.label_cooling_mode.setText("Max Hız")
            self.label_cooling_mode.setStyleSheet("color: #f38ba8; font-size: 11px; font-weight: bold;")
        elif mode == 1:
            self.label_cooling_mode.setText("Hızlı (Batch: 8)")
            self.label_cooling_mode.setStyleSheet("color: #f9e2af; font-size: 11px; font-weight: bold;")
        else:  # mode == 2
            self.label_cooling_mode.setText("Yavaş (Batch: 4)")
            self.label_cooling_mode.setStyleSheet("color: #a6e3a1; font-size: 11px; font-weight: bold;")
    
    def toggle_dl_settings(self):
        if self.rb_dl.isChecked():
            self.dl_widget.show()
            self.model_path.setText(os.path.join(os.getcwd(), "models", "physiognomy_net.pth"))
        else:
            self.dl_widget.hide()
            self.model_path.setText(os.path.join(os.getcwd(), "models", "face_personality_rf.joblib"))
            
    def browse_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Veri Dosyasını Seç", self.data_path.text(), "CSV Files (*.csv)", options=QFileDialog.DontUseNativeDialog)
        if path:
            self.data_path.setText(path)
            
    def browse_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Model Kayıt Yeri", self.model_path.text(), "Model Files (*.joblib *.pth)", options=QFileDialog.DontUseNativeDialog)
        if path:
            self.model_path.setText(path)
            
    def browse_test_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Images (*.jpg *.jpeg *.png)", options=QFileDialog.DontUseNativeDialog)
        if path:
            self.test_img_path.setText(path)
            
    def log(self, message):
        self.log_area.append(message)
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())
        
    def start_training(self):
        data_path = self.data_path.text()
        model_path = self.model_path.text()
        target_col = self.combo_target.currentText()
        
        if not os.path.exists(data_path):
            QMessageBox.warning(self, "Hata", f"Veri dosyası bulunamadı:\n{data_path}")
            return
            
        self.btn_train.setEnabled(False)
        self.progress.show()
        self.log("-" * 40)
        
        model_type = 'dl' if self.rb_dl.isChecked() else 'rf'
        backbone = self.combo_backbone.currentText()
        try:
            epochs = int(self.spin_epochs.text())
        except:
            epochs = 25
        
        # Calculate batch size and accumulation steps based on cooling mode
        if model_type == 'dl':
            cooling_mode = self.slider_cooling.value()
            if cooling_mode == 0:  # Auto (Max Hız)
                batch_size = 16
            elif cooling_mode == 1:  # Hızlı
                batch_size = 8
            else:  # cooling_mode == 2: Yavaş
                batch_size = 4
            
            # Calculate accumulation to maintain effective batch of 16
            target_effective_batch = 16
            accumulation_steps = max(1, target_effective_batch // batch_size)
        else:
            batch_size = 16
            accumulation_steps = 1
        
        self.thread = TrainingThread(data_path, model_path, target_col, model_type, backbone, epochs, batch_size, accumulation_steps)
        self.thread.log_signal.connect(self.log)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_training_finished)
        self.thread.start()
        
    def update_progress(self, percent):
        """Progress bar güncelle"""
        # Map 0-100 float to 0-10000 int
        val = int(percent * 100)
        self.progress.setValue(val)
        self.progress.setFormat(f"{percent:.2f}% - Eğitim İlerlemesi")
        
        # Calculate ETA
        if hasattr(self, 'thread') and hasattr(self.thread, 'start_time') and percent > 0:
            elapsed = time.time() - self.thread.start_time
            # percent is 0-100
            total_estimated = (elapsed / percent) * 100
            remaining = total_estimated - elapsed
            
            if remaining > 0:
                eta_str = str(timedelta(seconds=int(remaining)))
                self.progress.setFormat(f"{percent:.2f}% - Kalan Süre: {eta_str}")
        
    def on_training_finished(self, success, message):
        self.btn_train.setEnabled(True)
        self.progress.hide()
        if success:
            self.log(f"✅ {message}")
            QMessageBox.information(self, "Başarılı", message)
        else:
            self.log(f"❌ {message}")
            QMessageBox.critical(self, "Hata", message)
            
    def run_prediction(self):
        # Use selected model from combobox
        model_name = self.combo_test_model.currentText()
        if not model_name or model_name == "Model bulunamadı":
             QMessageBox.warning(self, "Hata", "Lütfen geçerli bir model seçin.")
             return
             
        current_model_path = os.path.join(os.getcwd(), "models", model_name)
             
        img_path = self.test_img_path.text()
        
        if not os.path.exists(current_model_path):
            QMessageBox.warning(self, "Hata", f"Model dosyası bulunamadı:\n{current_model_path}\n\nLütfen önce 'Model Eğitimi' sekmesinden bir model eğitin.")
            return
        if not os.path.exists(img_path):
            QMessageBox.warning(self, "Hata", "Lütfen geçerli bir resim seçin.")
            return
            
        self.result_area.setText("⏳ Analiz yapılıyor, lütfen bekleyin...")
        self.result_area.setStyleSheet("color: #f9e2af; font-size: 14px; background-color: #181825; padding: 20px; border-radius: 6px;")
        self.btn_predict.setEnabled(False)
        self.btn_predict.setText("⏳ İşleniyor...")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(base_dir, "scripts", "predict_image.py")
        
        cmd = [
            sys.executable, script_path,
            '--model', current_model_path,
            '--image', img_path
        ]
        
        self.pred_thread = PredictionThread(cmd)
        self.pred_thread.finished_signal.connect(self.on_prediction_success)
        self.pred_thread.error_signal.connect(self.on_prediction_error)
        self.pred_thread.finished.connect(lambda: self.on_prediction_complete())
        self.pred_thread.start()
        
    def on_prediction_success(self, output):
        # Pretty print result
        html = "<h3 style='color: #a6e3a1;'>✅ Analiz Tamamlandı</h3>"
        html += "<table style='width:100%;'>"
        
        if 'face_shape' in output:
            fs = output['face_shape']
            if isinstance(fs, dict):
                html += f"<tr><td style='color:#89b4fa;'><b>Yüz Şekli:</b></td><td style='color:#cdd6f4;'>{fs['prediction']}</td></tr>"
                html += f"<tr><td style='color:#89b4fa;'><b>Güven Skoru:</b></td><td style='color:#cdd6f4;'>{fs['confidence']:.2f}</td></tr>"
            else:
                html += f"<tr><td style='color:#89b4fa;'><b>Yüz Şekli:</b></td><td style='color:#cdd6f4;'>{fs}</td></tr>"
        
        html += "<tr><td colspan='2'><hr style='border: 1px solid #45475a;'></td></tr>"
        
        for k, v in output.items():
            if k not in ['face_shape', 'embedding_size', 'message']:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                html += f"<tr><td style='color:#a6adc8;'>{k}:</td><td style='color:#cdd6f4;'>{val_str}</td></tr>"
                
        html += "</table>"
        
        if 'message' in output:
             html += f"<p style='color:#f38ba8; font-size:11px;'><i>Not: {output['message']}</i></p>"
        
        self.result_area.setText(html)
        self.result_area.setStyleSheet("background-color: #181825; border-radius: 6px; padding: 20px;")
        
    def on_prediction_error(self, error_msg):
        self.result_area.setText(f"❌ {error_msg}")
        self.result_area.setStyleSheet("color: #f38ba8; font-size: 14px; background-color: #181825; padding: 20px; border-radius: 6px;")
        
    def on_prediction_complete(self):
        self.btn_predict.setEnabled(True)
        self.btn_predict.setText("🔮 Tahmin Et")

    def cleanup(self):
        """Stop any running training threads"""
        if hasattr(self, 'thread') and self.thread is not None:
            try:
                if self.thread.isRunning():
                    self.thread.stop()
                    self.thread.quit()
                    self.thread.wait()
            except (RuntimeError, AttributeError):
                # Thread might be deleted or in invalid state
                pass
