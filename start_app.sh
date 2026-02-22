#!/bin/bash

echo "🚀 Fizyonomi AI Web Uygulaması Başlatılıyor..."
echo ""
echo "🐍 Python 3.11 + MediaPipe"
echo "📋 Gerekli dizinler kontrol ediliyor..."
mkdir -p archive/photos archive/results templates static/css static/js

echo "✅ Hazır!"
echo ""
echo "🌐 Uygulama başlatılıyor..."
echo "📱 Tarayıcınızda açın: http://localhost:5000"
echo ""
echo "⚠️  Durdurmak için Ctrl+C tuşlayın"
echo ""

/Volumes/Data/workspace/yasin/.venv-py311/bin/python app.py
