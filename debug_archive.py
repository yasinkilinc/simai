#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug script to check archive database entries and file paths"""
import sqlite3
import os
import sys

# Add desktop_app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'desktop_app'))

from database import Database

# Initialize database
db = Database()

# Get recent analyses
print("=== Veritabanındaki Son Analizler ===\n")
analyses = db.get_recent_analyses()

if not analyses:
    print("⚠️  Veritabanında hiç kayıt yok!")
else:
    print(f"Toplam {len(analyses)} kayıt bulundu:\n")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    archive_dataset_root = os.path.join(base_dir, "dataset", "archive")
    archive_root = os.path.join(base_dir, "archive")
    
    for idx, row in enumerate(analyses, 1):
        print(f"{idx}. Analiz:")
        print(f"   ID: {row['id']}")
        print(f"   Filename (DB): {row['filename']}")
        print(f"   Face Shape: {row['face_shape']}")
        print(f"   Timestamp: {row['timestamp']}\n")
        
        filename = row['filename']
        
        # Check new structure
        new_path = os.path.join(archive_dataset_root, filename, "front.jpg")
        print(f"   Yeni Yapı Yolu: {new_path}")
        print(f"   Dosya Var mı? {'✅ EVET' if os.path.exists(new_path) else '❌ YOK'}")
        
        # Check old structure
        old_path = os.path.join(archive_root, "images", filename)
        print(f"   Eski Yapı Yolu: {old_path}")
        print(f"   Dosya Var mı? {'✅ EVET' if os.path.exists(old_path) else '❌ YOK'}")
        
        print()

print("\n=== Klasör Yapısı ===\n")
print(f"Dataset/Archive Klasörü: {archive_dataset_root}")
if os.path.exists(archive_dataset_root):
    folders = [f for f in os.listdir(archive_dataset_root) if os.path.isdir(os.path.join(archive_dataset_root, f))]
    print(f"Klasörler: {folders}")
else:
    print("❌ Klasör bulunamadı!")

print(f"\nArchive Klasörü: {archive_root}")
if os.path.exists(archive_root):
    items = os.listdir(archive_root)
    print(f"İçerik: {items}")
else:
    print("❌ Klasör bulunamadı!")
