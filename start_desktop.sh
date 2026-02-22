#!/bin/bash

echo "🚀 Fizyonomi AI Desktop Uygulaması"
echo "=================================="
echo ""
echo "🐍 Python 3.11 + MediaPipe"
echo "📋 Log dosyası: logs/app.log"
echo "   Tüm çıktılar kaydediliyor..."
echo ""
echo "⚠️  Hata olursa 'logs/app.log' dosyasını kontrol edin!"
echo ""
echo "🎬 Başlatılıyor..."
echo ""

cd "$(dirname "$0")"

# Python 3.11 environment kullan (MediaPipe desteği ile)
/Volumes/Data/workspace/yasin/.venv-py311/bin/python desktop_app/main.py 2>&1
