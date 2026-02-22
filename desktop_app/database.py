import sqlite3
import json
import os
from datetime import datetime
import cv2
import numpy as np

class Database:
    def __init__(self, db_path=None):
        if db_path is None:
            # Use absolute path based on this file's location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "db", "fizyonomi.db")
            
        self.db_path = db_path
        self.ensure_db()

    def ensure_db(self):
        """Veritabanı ve tabloları oluştur"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Photos table with BLOB support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                face_shape TEXT,
                
                -- Images (BLOB)
                front_image BLOB NOT NULL,
                side_image BLOB,
                heatmap_image BLOB,
                
                -- Data (JSON)
                analysis_json TEXT,
                points_3d_json TEXT,
                landmarks_json TEXT,
                
                -- Metadata
                image_width INTEGER,
                image_height INTEGER
            )
        """)
        
        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def ensure_db(self):
        """Veritabanı ve tabloları oluştur"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Photos table with BLOB support
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                face_shape TEXT,
                
                -- Images (BLOB)
                front_image BLOB NOT NULL,
                side_image BLOB,
                heatmap_image BLOB,
                
                -- Data (JSON)
                analysis_json TEXT,
                points_3d_json TEXT,
                landmarks_json TEXT,
                
                -- Metadata
                image_width INTEGER,
                image_height INTEGER
            )
        """)
        
        # Settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Annotations table (Training Data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT UNIQUE NOT NULL,
                image_data BLOB NOT NULL,
                annotations_json TEXT,
                split TEXT DEFAULT 'train', -- train, val, test
                status TEXT DEFAULT 'pending', -- pending, annotated, skipped
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def save_training_data(self, image_name, image_data, split='train'):
        """Eğitim verisi olarak resim kaydet (henüz etiketlenmemiş)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert image to BLOB if needed
        if isinstance(image_data, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image_data)
            blob = buffer.tobytes()
        else:
            blob = image_data

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO annotations (image_name, image_data, split, status)
                VALUES (?, ?, ?, 'pending')
            """, (image_name, blob, split))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error saving training data: {e}")
            return None
        finally:
            conn.close()

    def get_next_pending_annotation(self):
        """Sıradaki etiketlenmemiş resmi getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM annotations 
            WHERE status = 'pending' 
            ORDER BY id ASC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            # Convert BLOB to numpy
            nparr = np.frombuffer(result['image_data'], np.uint8)
            result['image'] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return result
        return None

    def update_annotation(self, image_name, annotations):
        """Etiketleri kaydet ve durumu güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE annotations 
            SET annotations_json = ?, status = 'annotated', updated_at = CURRENT_TIMESTAMP
            WHERE image_name = ?
        """, (json.dumps(annotations, ensure_ascii=False), image_name))
        
        conn.commit()
        conn.close()

    def get_annotation_status(self):
        """Etiketleme durumunu getir (toplam, etiketlenen)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM annotations")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM annotations WHERE status = 'annotated'")
        annotated = cursor.fetchone()[0]
        
        conn.close()
        return {'total': total, 'annotated': annotated}

    def get_all_annotation_ids(self):
        """Tüm etiketleme ID'lerini getir (sıralı)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM annotations ORDER BY id ASC")
        rows = cursor.fetchall()
        
        conn.close()
        return [row[0] for row in rows]

    def get_annotation_by_id(self, annotation_id):
        """ID'ye göre etiketleme verisini getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            result = dict(row)
            # Convert BLOB to numpy
            if result['image_data']:
                nparr = np.frombuffer(result['image_data'], np.uint8)
                result['image'] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Parse JSON
            if result['annotations_json']:
                result['annotations'] = json.loads(result['annotations_json'])
            else:
                result['annotations'] = None
                
            return result
        return None

    def get_all_annotations(self):
        """Tüm etiketlenmiş verileri getir"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, image_name, annotations_json, status 
            FROM annotations 
            WHERE status = 'annotated'
            ORDER BY id ASC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSON
            if result['annotations_json']:
                result['annotations'] = json.loads(result['annotations_json'])
            else:
                result['annotations'] = {}
            results.append(result)
        
        return results

    def save_setting(self, key, value):
        """Tek bir ayarı kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, str(value)))
        
        conn.commit()
        conn.close()
    
    def save_settings_batch(self, settings_dict):
        """Birden fazla ayarı toplu kaydet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for key, value in settings_dict.items():
            cursor.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, str(value)))
        
        conn.commit()
        conn.close()
    
    def get_setting(self, key, default=None):
        """Tek bir ayarı getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else default
    
    def get_all_settings(self, as_dict=True):
        """Tüm ayarları getir"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        conn.close()
        
        if as_dict:
            return {row[0]: row[1] for row in rows}
        return rows

    def save_analysis(self, front_image, face_shape, analysis_report, points_3d, landmarks, 
                     side_image=None, heatmap_image=None):
        """
        Analiz sonucunu BLOB olarak kaydet
        
        Args:
            front_image: numpy array (BGR)
            face_shape: str
            analysis_report: dict
            points_3d: numpy array or list
            landmarks: list of dicts
            side_image: numpy array (BGR) optional
            heatmap_image: numpy array (BGR) optional
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert images to BLOB (encode as JPEG)
        def img_to_blob(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                _, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes()
            return None
        
        front_blob = img_to_blob(front_image)
        side_blob = img_to_blob(side_image)
        heatmap_blob = img_to_blob(heatmap_image)
        
        # Get image dimensions
        if isinstance(front_image, np.ndarray):
            height, width = front_image.shape[:2]
        else:
            width, height = 0, 0
        
        # Convert numpy arrays to list for JSON serialization
        points_list = points_3d.tolist() if hasattr(points_3d, 'tolist') else points_3d
        landmarks_list = landmarks if isinstance(landmarks, list) else landmarks.tolist() if hasattr(landmarks, 'tolist') else []
        
        cursor.execute("""
            INSERT INTO photos (
                front_image, side_image, heatmap_image,
                face_shape, analysis_json, points_3d_json, landmarks_json,
                image_width, image_height
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            front_blob,
            side_blob,
            heatmap_blob,
            face_shape,
            json.dumps(analysis_report, ensure_ascii=False),
            json.dumps(points_list),
            json.dumps(landmarks_list),
            width,
            height
        ))
        
        photo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return photo_id

    def update_analysis(self, photo_id, front_image, face_shape, analysis_report, points_3d, landmarks,
                       side_image=None, heatmap_image=None):
        """Mevcut analizi güncelle"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert images to BLOB
        def img_to_blob(img):
            if img is None:
                return None
            if isinstance(img, np.ndarray):
                _, buffer = cv2.imencode('.jpg', img)
                return buffer.tobytes()
            return None
        
        front_blob = img_to_blob(front_image)
        side_blob = img_to_blob(side_image)
        heatmap_blob = img_to_blob(heatmap_image)
        
        # Get image dimensions
        if isinstance(front_image, np.ndarray):
            height, width = front_image.shape[:2]
        else:
            width, height = 0, 0
        
        # Convert numpy arrays
        points_list = points_3d.tolist() if hasattr(points_3d, 'tolist') else points_3d
        landmarks_list = landmarks if isinstance(landmarks, list) else landmarks.tolist() if hasattr(landmarks, 'tolist') else []
        
        print(f"DEBUG: update_analysis. landmarks type: {type(landmarks)}")
        print(f"DEBUG: update_analysis. landmarks_list len: {len(landmarks_list)}")
        
        cursor.execute("""
            UPDATE photos 
            SET front_image = ?, side_image = ?, heatmap_image = ?,
                face_shape = ?, analysis_json = ?, points_3d_json = ?, landmarks_json = ?,
                image_width = ?, image_height = ?,
                timestamp = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            front_blob,
            side_blob,
            heatmap_blob,
            face_shape,
            json.dumps(analysis_report, ensure_ascii=False),
            json.dumps(points_list),
            json.dumps(landmarks_list),
            width,
            height,
            photo_id
        ))
        
        conn.commit()
        conn.close()

    def get_recent_analyses(self, limit=10):
        """Son analizleri getir (sadece metadata, BLOB olmadan)"""
        print(f"DEBUG: Database.get_recent_analyses called. DB Path: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, face_shape, image_width, image_height
            FROM photos 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        print(f"DEBUG: Database query returned {len(rows)} rows")
        conn.close()
        return [dict(row) for row in rows]
    
    def get_analysis_by_id(self, photo_id):
        """Belirli bir analizi tüm verileriyle getir (BLOB dahil)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM photos WHERE id = ?
        """, (photo_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        result = dict(row)
        
        # Convert BLOBs to numpy arrays
        def blob_to_img(blob):
            if blob is None:
                return None
            nparr = np.frombuffer(blob, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result['front_image'] = blob_to_img(result['front_image'])
        result['side_image'] = blob_to_img(result['side_image'])
        result['heatmap_image'] = blob_to_img(result['heatmap_image'])
        
        # Parse JSON fields
        if result['analysis_json']:
            result['analysis'] = json.loads(result['analysis_json'])
        if result['points_3d_json']:
            result['points_3d'] = json.loads(result['points_3d_json'])
        if result['landmarks_json']:
            result['landmarks'] = json.loads(result['landmarks_json'])
        
        return result

    def delete_analysis(self, photo_id):
        """Analizi sil"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
        
        conn.commit()
        conn.close()
