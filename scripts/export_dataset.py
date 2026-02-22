import os
import sys
import sqlite3
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desktop_app.database import Database

def export_dataset(output_dir="dataset/export"):
    """
    Export annotated data from database to a format suitable for ML training.
    Creates:
    - output_dir/images/ (raw images)
    - output_dir/data.csv (features and labels)
    - output_dir/landmarks.json (raw landmarks for DL)
    """
    print(f"Starting export to {output_dir}...")
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Connect to DB
    db = Database()
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Select only annotated records
    cursor.execute("SELECT * FROM annotations WHERE status = 'annotated'")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} annotated records.")
    
    data_list = []
    landmarks_dict = {}
    
    for row in rows:
        record = dict(row)
        image_name = record['image_name']
        
        # 1. Save Image
        if record['image_data']:
            image_path = os.path.join(images_dir, image_name)
            # Only write if not exists to save time, or overwrite? Let's overwrite to be safe.
            with open(image_path, 'wb') as f:
                f.write(record['image_data'])
        
        # 2. Process Annotations & Metrics
        if not record['annotations_json']:
            print(f"Skipping {image_name}: No annotation JSON")
            continue
            
        try:
            annotations = json.loads(record['annotations_json'])
            
            # Save Landmarks if available
            if 'landmarks' in annotations:
                # annotations['landmarks'] is usually a list of dicts or list of lists
                # We need list of [x, y, z] or [x, y]
                # Check format
                lms = annotations['landmarks']
                # If it's the format from AutoAnnotator (list of {x, y, z}), convert to list of lists
                if lms and isinstance(lms[0], dict):
                    lms_list = [[p['x'], p['y'], p.get('z', 0)] for p in lms]
                else:
                    lms_list = lms
                
                landmarks_dict[image_name] = lms_list
            
            # Base row data
            csv_row = {
                'image_name': image_name,
                'split': record['split'],
                'timestamp': record['timestamp']
            }
            
            # Add Metrics (Features)
            if 'metrics' in annotations:
                for key, value in annotations['metrics'].items():
                    csv_row[f"feat_{key}"] = value
            else:
                # print(f"Warning: No metrics found for {image_name}.")
                pass
            
            # Add Labels (Targets)
            if 'face_shape' in annotations:
                csv_row['target_face_shape'] = annotations['face_shape'].get('shape')
                
            if 'forehead' in annotations:
                csv_row['target_forehead_width'] = annotations['forehead'].get('width')
                csv_row['target_forehead_height'] = annotations['forehead'].get('height')
                csv_row['target_forehead_slope'] = annotations['forehead'].get('slope')
                
            if 'eyes' in annotations:
                csv_row['target_eyes_size'] = annotations['eyes'].get('size')
                csv_row['target_eyes_slant'] = annotations['eyes'].get('slant')
                csv_row['target_eyes_spacing'] = annotations['eyes'].get('spacing')
                csv_row['target_eyes_depth'] = annotations['eyes'].get('depth')
                
            if 'nose' in annotations:
                csv_row['target_nose_length'] = annotations['nose'].get('length')
                csv_row['target_nose_width'] = annotations['nose'].get('width')
                csv_row['target_nose_bridge'] = annotations['nose'].get('bridge')
                csv_row['target_nose_tip'] = annotations['nose'].get('tip')
                
            if 'lips' in annotations:
                csv_row['target_lips_upper'] = annotations['lips'].get('upper_thickness')
                csv_row['target_lips_lower'] = annotations['lips'].get('lower_thickness')
                csv_row['target_lips_width'] = annotations['lips'].get('width')
                
            if 'chin' in annotations:
                csv_row['target_chin_width'] = annotations['chin'].get('width')
                csv_row['target_chin_prominence'] = annotations['chin'].get('prominence')
                csv_row['target_chin_dimple'] = annotations['chin'].get('dimple')
                
            if 'ears' in annotations:
                csv_row['target_ears_size'] = annotations['ears'].get('size')
                csv_row['target_ears_prominence'] = annotations['ears'].get('prominence')
                csv_row['target_ears_lobe'] = annotations['ears'].get('lobe')
            
            data_list.append(csv_row)
            
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {image_name}")
            continue
            
    conn.close()
    
    # Save Landmarks JSON
    lm_path = os.path.join(output_dir, "landmarks.json")
    with open(lm_path, 'w') as f:
        json.dump(landmarks_dict, f)
    print(f"Landmarks saved to: {lm_path} ({len(landmarks_dict)} records)")
    
    # Create DataFrame and Save
    if data_list:
        df = pd.DataFrame(data_list)
        csv_path = os.path.join(output_dir, "data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Export completed successfully.")
        print(f"Data saved to: {csv_path}")
        print(f"Images saved to: {images_dir}")
        print(f"Total records: {len(df)}")
        
        # Print column info
        feat_cols = [c for c in df.columns if c.startswith('feat_')]
        target_cols = [c for c in df.columns if c.startswith('target_')]
        print(f"Features: {len(feat_cols)}")
        print(f"Targets: {len(target_cols)}")
    else:
        print("No data exported.")

if __name__ == "__main__":
    export_dataset()
