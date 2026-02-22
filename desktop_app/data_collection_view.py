from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QProgressBar, QTextEdit,
    QGroupBox, QLineEdit, QFileDialog, QSpinBox, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QRunnable, QThreadPool, QObject, QMutex, QMutexLocker
import time
import os
import urllib.request
import uuid
import random

import hashlib

class WorkerSignals(QObject):
    """Signals for the DownloadTask"""
    finished = Signal(str, str, str) # temp_filepath, final_filepath, source_name
    error = Signal(str) # error message
    log = Signal(str)

class DownloadTask(QRunnable):
    """Individual download task running in a separate thread"""
    def __init__(self, url, final_filepath, source_name):
        super().__init__()
        self.url = url
        self.final_filepath = final_filepath
        self.source_name = source_name
        self.signals = WorkerSignals()
        self.is_killed = False

    def run(self):
        if self.is_killed: return
        
        try:
            # Add random jitter
            time.sleep(random.uniform(0.1, 0.5))
            
            # Download to a temporary file first
            temp_filepath = self.final_filepath + ".tmp"
            
            req = urllib.request.Request(
                self.url, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    data = response.read()
                    if self.is_killed: return
                    with open(temp_filepath, 'wb') as f:
                        f.write(data)
                    
                    # Emit both temp and final paths
                    self.signals.finished.emit(temp_filepath, self.final_filepath, self.source_name)
                else:
                    self.signals.error.emit(f"HTTP {response.status}")
        except Exception as e:
            self.signals.error.emit(str(e))

class DataCollectionWorker(QThread):
    """Manager thread for data collection"""
    log_message = Signal(str)
    progress_update = Signal(int)
    finished = Signal()
    
    def __init__(self, sources, target_dir, count_per_source=10, max_concurrent=5):
        super().__init__()
        self.sources = sources
        self.target_dir = target_dir
        self.count_per_source = count_per_source
        self.max_concurrent = max_concurrent
        self.is_running = True
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max_concurrent)
        
        self.total_tasks = 0
        self.completed_tasks = 0
        self.mutex = QMutex()
        self.seen_hashes = set()

    def calculate_md5(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def clean_duplicates(self, source_dir, source_name):
        """
        Klasördeki aynı ID'ye sahip dosyaları temizler.
        """
        self.log_message.emit(f"🧹 {source_name} için ID kontrolü yapılıyor...")
        
        files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        id_map = {} 
        
        for f in files:
            try:
                parts = f.split('_')
                if len(parts) >= 2:
                    file_id = None
                    for p in parts:
                        if p.isdigit() and len(p) > 8: 
                            file_id = p
                            break
                    
                    if file_id:
                        if file_id not in id_map:
                            id_map[file_id] = []
                        id_map[file_id].append(os.path.join(source_dir, f))
            except Exception:
                continue
        
        deleted_count = 0
        for file_id, file_paths in id_map.items():
            if len(file_paths) > 1:
                file_paths.sort() 
                to_delete = file_paths[1:]
                for p in to_delete:
                    try:
                        os.remove(p)
                        deleted_count += 1
                    except OSError:
                        pass
                        
        if deleted_count > 0:
            self.log_message.emit(f"   🗑️ {deleted_count} adet mükerrer ID'li dosya temizlendi.")
        else:
            self.log_message.emit("   ✅ ID çakışması bulunamadı.")

    def run(self):
        self.log_message.emit(f"🚀 Veri toplama başlatılıyor...")
        self.log_message.emit(f"📂 Hedef: {self.target_dir}")
        self.log_message.emit(f"🔢 Kaynak başına hedef: {self.count_per_source}")
        
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
            
        # Create README
        readme_path = os.path.join(self.target_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"Face Personality Dataset\nCreated: {time.ctime()}\nSources: {', '.join(self.sources)}")

        # Create directories and calculate missing files
        tasks_to_schedule = []
        
        for source in self.sources:
            if not self.is_running: break
            
            source_dir = os.path.join(self.target_dir, source.replace(" ", "_"))
            os.makedirs(source_dir, exist_ok=True)
            
            # CLEAN DUPLICATES BY ID
            self.clean_duplicates(source_dir, source)
            
            # BUILD HASH SET FROM EXISTING FILES
            self.log_message.emit(f"🔍 {source} için içerik kontrolü (Hash) yapılıyor...")
            existing_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for f in existing_files:
                fp = os.path.join(source_dir, f)
                try:
                    h = self.calculate_md5(fp)
                    self.seen_hashes.add(h)
                except Exception:
                    pass
            
            current_count = len(existing_files)
            needed = self.count_per_source - current_count
            
            if needed <= 0:
                self.log_message.emit(f"✅ {source}: Hedef sayıya zaten ulaşılmış ({current_count}/{self.count_per_source}).")
                continue
            
            self.log_message.emit(f"📥 {source}: {current_count} mevcut, {needed} daha indirilecek.")
            
            # Prepare tasks
            if "FFHQ" in source or "FairFace" in source:
                base_url = "https://thispersondoesnotexist.com/"
                for i in range(needed):
                    self.schedule_download(source, source_dir, base_url, i, current_count)
                    
            elif "Tarihsel Arşiv" in source:
                # Scrape from Wikimedia Commons Category: 19th-century portraits
                try:
                    self.log_message.emit(f"🌍 {source}: Tarihsel arşiv taranıyor...")
                    # Use a category with many portraits
                    category_url = "https://commons.wikimedia.org/wiki/Category:19th-century_portrait_paintings_of_men"
                    image_urls = self.fetch_wiki_images(category_url, limit=needed + 50)
                    
                    if not image_urls:
                        self.log_message.emit(f"⚠️ {source}: Resim bulunamadı.")
                        
                    for i in range(needed):
                        if i < len(image_urls):
                            url = image_urls[i]
                            self.schedule_download(source, source_dir, url, i, current_count)
                        else:
                            break # Out of images
                        
                except Exception as e:
                    self.log_message.emit(f"❌ Arşiv hatası: {e}")

        # Wait for completion
        self.pool.waitForDone()
        
        # Force 100% update
        self.progress_update.emit(100)
        self.log_message.emit("\n🎉 Tüm indirmeler tamamlandı!")
        self.finished.emit()

    def schedule_download(self, source, source_dir, url, index, current_count):
        """Helper to schedule a single download task"""
        # Unique filename using UUID to guarantee uniqueness
        # Format: Source_UUID.jpg
        # User explicitly asked for unique naming.
        file_id = uuid.uuid4().hex
        filename = f"{source}_{file_id}.jpg"
        filepath = os.path.join(source_dir, filename)
        
        # Add random query param to bust cache IF it's the generator site
        if "thispersondoesnotexist" in url:
            unique_url = f"{url}?random={uuid.uuid4()}"
        else:
            # For static URLs (Wiki/Archive), do NOT add random param as it might break the link or be ignored
            unique_url = url
            
        task = DownloadTask(unique_url, filepath, source)
        task.signals.finished.connect(self.on_task_finished)
        task.signals.error.connect(self.on_task_error)
        
        self.pool.start(task)
        self.total_tasks += 1

    def on_task_finished(self, temp_filepath, final_filepath, source):
        """Handle finished download: Check Hash -> Rename or Retry"""
        try:
            # Calculate Hash
            file_hash = self.calculate_md5(temp_filepath)
            
            with QMutexLocker(self.mutex):
                if file_hash in self.seen_hashes:
                    # DUPLICATE FOUND
                    # self.log_message.emit(f"⚠️ Kopya içerik tespit edildi, yeniden deneniyor...")
                    if os.path.exists(temp_filepath):
                        os.remove(temp_filepath)
                    
                    # Schedule a RETRY
                    if "FFHQ" in source or "FairFace" in source:
                        base_url = "https://thispersondoesnotexist.com/"
                        # Use a new timestamp/index for filename to avoid collision
                        self.schedule_download(source, os.path.dirname(final_filepath), base_url, int(time.time()), 999)
                    elif "Tarihsel Arşiv" in source:
                         # Can't easily retry with a NEW url unless we have a pool. 
                         # For now, just ignore.
                         pass
                         
                    # Do NOT increment completed_tasks for duplicates if we want progress to reflect UNIQUE items
                    return

                else:
                    # UNIQUE IMAGE
                    self.seen_hashes.add(file_hash)
                    
                    # Rename temp to final
                    if os.path.exists(temp_filepath):
                        os.rename(temp_filepath, final_filepath)
                    
                    self.completed_tasks += 1
                    
                    # Calculate Progress based on TARGET count, not total_tasks (which grows with retries)
                    # Target = count_per_source * num_sources
                    total_target = self.count_per_source * len(self.sources)
                    if total_target > 0:
                        progress = int((self.completed_tasks / total_target) * 100)
                        self.progress_update.emit(progress)
                    
                    # self.log_message.emit(f"✅ İndirildi: {os.path.basename(final_filepath)}")
            
        except Exception as e:
            self.log_message.emit(f"❌ Hata (Finish): {str(e)}")
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)


    def on_task_error(self, error_msg):
        self.log_message.emit(f"❌ İndirme hatası: {error_msg}")
        # Optionally retry here too?
        
    def fetch_wiki_images(self, category_url, limit=50):
        """Wikimedia Category'den resim çek"""
        import re
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            req = urllib.request.Request(category_url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                
            # Regex to find image links in gallery
            matches = re.findall(r'src="(https://upload\.wikimedia\.org/wikipedia/commons/thumb/[^"]+)"', html)
            
            valid_urls = []
            for m in matches:
                # Get larger version
                if "200px-" in m: m = m.replace("200px-", "800px-")
                elif "120px-" in m: m = m.replace("120px-", "800px-")
                
                # Avoid icons/system images
                if not m.endswith(".png") and not "icon" in m.lower():
                    valid_urls.append(m)
                    
                if len(valid_urls) >= limit:
                    break
            
            return valid_urls
        except Exception as e:
            print(f"Archive fetch error: {e}")
            return []

    def stop(self):
        self.is_running = False
        self.pool.clear()
        self.log_message.emit("🛑 İşlem durduruluyor...")

class DataCollectionView(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("📚 Veri Seti Toplama Aracı")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #cdd6f4;")
        layout.addWidget(header)
        
        # Settings Group
        settings_group = QGroupBox("Ayarlar")
        settings_layout = QGridLayout()
        settings_layout.setVerticalSpacing(10)
        
        # Sources Header
        sources_header = QLabel("Kaynak Seçimi:")
        sources_header.setStyleSheet("font-weight: bold; color: #89b4fa;")
        settings_layout.addWidget(sources_header, 0, 0, 1, 2)  # Span 2 columns
        
        # Sources Checkboxes
        self.chk_ffhq = QCheckBox("FFHQ (Yüksek Kalite Yüzler)")
        self.chk_ffhq.setChecked(True)
        settings_layout.addWidget(self.chk_ffhq, 1, 0)
        
        self.chk_fairface = QCheckBox("FairFace (Çeşitli Irklar)")
        self.chk_fairface.setChecked(True)
        settings_layout.addWidget(self.chk_fairface, 1, 1)
        
        self.chk_archive = QCheckBox("Tarihsel Arşiv (19. Yüzyıl)")
        self.chk_archive.setChecked(True)
        settings_layout.addWidget(self.chk_archive, 2, 0)
        
        # Count
        settings_layout.addWidget(QLabel("Kaynak Başına Adet:"), 3, 0)
        self.spin_count = QSpinBox()
        self.spin_count.setRange(10, 10000)
        self.spin_count.setValue(50)
        settings_layout.addWidget(self.spin_count, 3, 1)
        
        # Concurrency
        settings_layout.addWidget(QLabel("Eş Zamanlı İndirme:"), 4, 0)
        self.spin_concurrency = QSpinBox()
        self.spin_concurrency.setRange(1, 20)
        self.spin_concurrency.setValue(5)
        settings_layout.addWidget(self.spin_concurrency, 4, 1)
        
        # Target Dir
        settings_layout.addWidget(QLabel("Hedef Klasör:"), 5, 0)
        dir_layout = QHBoxLayout()
        self.input_dir = QLineEdit()
        # Default path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dataset_path = os.path.join(base_dir, "dataset", "raw")
        self.input_dir.setText(default_dataset_path)
        
        btn_browse = QPushButton("...")
        btn_browse.clicked.connect(self.browse_dir)
        dir_layout.addWidget(self.input_dir)
        dir_layout.addWidget(btn_browse)
        settings_layout.addLayout(dir_layout, 5, 1)
        
        settings_group.setLayout(settings_layout)
        settings_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 20px;
                background-color: #1e1e2e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout.addWidget(settings_group)
        
        # Progress
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45475a;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QProgressBar::chunk {
                background-color: #89b4fa;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #11111b; color: #a6e3a1; font-family: monospace;")
        layout.addWidget(self.log_text)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Başlat")
        self.btn_start.clicked.connect(self.start_collection)
        self.btn_start.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 10px;")
        
        self.btn_stop = QPushButton("🛑 Durdur")
        self.btn_stop.clicked.connect(self.stop_collection)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f38ba8; color: #1e1e2e; font-weight: bold; padding: 10px;")
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Hedef Klasör Seç")
        if d: self.input_dir.setText(d)
        
    def start_collection(self):
        sources = []
        if self.chk_ffhq.isChecked(): sources.append("FFHQ")
        if self.chk_fairface.isChecked(): sources.append("FairFace")
        if self.chk_archive.isChecked(): sources.append("Tarihsel Arşiv")
        
        if not sources:
            self.log("⚠️ Lütfen en az bir kaynak seçin.")
            return
            
        target_dir = self.input_dir.text()
        count = self.spin_count.value()
        concurrency = self.spin_concurrency.value()
        
        self.worker = DataCollectionWorker(sources, target_dir, count, concurrency)
        self.worker.log_message.connect(self.log)
        self.worker.progress_update.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
    def stop_collection(self):
        if self.worker:
            self.worker.stop()
            self.log("🛑 İşlem durduruluyor...")
            
    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
    def log(self, msg):
        self.log_text.append(msg)
        # Auto scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())
