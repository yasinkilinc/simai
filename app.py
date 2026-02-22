from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import sys
import cv2
import numpy as np
import json
import sqlite3
from datetime import datetime
import base64

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from reconstruction import FaceReconstructor
from features import FaceFeatures
from interpreter import PhysiognomyInterpreter
from visualizer import Visualizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'archive/photos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = 'ilmisima_gizli_anahtar_super_guvenli'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('archive/results', exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('archive/database.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Photos table
    c.execute('''CREATE TABLE IF NOT EXISTS photos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  face_detected BOOLEAN,
                  face_shape TEXT,
                  analysis_json TEXT)''')
                  
    # Migrate photos table (add user_id)
    try:
        c.execute('ALTER TABLE photos ADD COLUMN user_id INTEGER REFERENCES users(id)')
    except sqlite3.OperationalError:
        pass # Column already exists
    
    # Feedback table for learning
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  photo_id INTEGER,
                  trait TEXT,
                  user_rating INTEGER,
                  comment TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (photo_id) REFERENCES photos(id))''')
    
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('archive/database.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                      (username, generate_password_hash(password)))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Bu kullanıcı adı alınmış.')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = sqlite3.connect('archive/database.db')
        c = conn.cursor()
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Kullanıcı adı veya şifre hatalı.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('archive/database.db')
    c = conn.cursor()
    c.execute('''SELECT id, filename, timestamp, face_shape, analysis_json 
                 FROM photos WHERE user_id = ? ORDER BY timestamp DESC''', (session['user_id'],))
    rows = c.fetchall()
    conn.close()
    
    history_data = []
    for r in rows:
        history_data.append({
            'photo_id': r[0],
            'filename': r[1],
            'timestamp': r[2],
            'face_shape': r[3],
            'analysis': json.loads(r[4]) if r[4] else {}
        })
        
    return render_template('history.html', history=history_data)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save to archive
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to load image'}), 400
        
        # Process
        reconstructor = FaceReconstructor()
        landmarks = reconstructor.process_frame(img)
        
        if not landmarks:
            # Save to DB without analysis
            save_to_db(filename, False, None, None, session.get('user_id'))
            return jsonify({'error': 'No face detected', 'photo_id': get_last_photo_id()}), 404
        
        # Extract 3D points
        points_3d = reconstructor.get_3d_points(landmarks, img.shape)
        
        # Extract features
        features = FaceFeatures(points_3d, img)
        
        # Interpret
        interpreter = PhysiognomyInterpreter()
        report = interpreter.interpret(features)
        
        # Visualize
        visualizer = Visualizer()
        annotated = img.copy()
        visualizer.draw_landmarks(annotated, points_3d)
        visualizer.draw_analysis(annotated, report)
        
        result_filename = f"result_{filename}"
        result_path = os.path.join('archive/results', result_filename)
        visualizer.save_image(annotated, result_path)
        
        # Save 3D mesh data (simplified)
        mesh_data = {
            'vertices': points_3d.tolist(),
            'landmarks_count': len(points_3d)
        }
        
        # Save to database
        user_id = session.get('user_id')
        photo_id = save_to_db(filename, True, report['face_shape'], json.dumps(report), user_id)
        
        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'result_image': f'/results/{result_filename}',
            'mesh_data': mesh_data,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for learning"""
    try:
        data = request.json
        photo_id = data.get('photo_id')
        trait = data.get('trait')
        rating = data.get('rating')  # 1-5 scale
        comment = data.get('comment', '')
        
        conn = sqlite3.connect('archive/database.db')
        c = conn.cursor()
        c.execute('''INSERT INTO feedback (photo_id, trait, user_rating, comment)
                     VALUES (?, ?, ?, ?)''', (photo_id, trait, rating, comment))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """Get analysis history"""
    try:
        conn = sqlite3.connect('archive/database.db')
        c = conn.cursor()
        c.execute('''SELECT id, filename, timestamp, face_detected, face_shape 
                     FROM photos ORDER BY timestamp DESC LIMIT 50''')
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'filename': row[1],
                'timestamp': row[2],
                'face_detected': bool(row[3]),
                'face_shape': row[4]
            })
        
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve result images"""
    return send_file(os.path.join('archive/results', filename))

@app.route('/photos/<filename>')
def serve_photo(filename):
    """Serve archived photos"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

def save_to_db(filename, detected, shape, analysis_json, user_id=None):
    """Save analysis to database"""
    conn = sqlite3.connect('archive/database.db')
    c = conn.cursor()
    c.execute('''INSERT INTO photos (filename, face_detected, face_shape, analysis_json, user_id)
                 VALUES (?, ?, ?, ?, ?)''', (filename, detected, shape, analysis_json, user_id))
    photo_id = c.lastrowid
    conn.commit()
    conn.close()
    return photo_id

def get_last_photo_id():
    """Get last inserted photo ID"""
    conn = sqlite3.connect('archive/database.db')
    c = conn.cursor()
    c.execute('SELECT MAX(id) FROM photos')
    photo_id = c.fetchone()[0]
    conn.close()
    return photo_id

if __name__ == '__main__':
    print("🚀 Fizyonomi Analiz Uygulaması Başlatılıyor...")
    print("📱 Tarayıcınızda açın: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
