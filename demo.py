import os
import subprocess
import urllib.request

def download_sample_image(url, filename):
    print(f"Örnek resim indiriliyor: {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print("İndirme başarılı.")
        return True
    except Exception as e:
        print(f"İndirme hatası: {e}")
        return False

def main():
    # Lena (Standard CV Sample)
    image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    image_filename = "sample_face.jpg"

    if not os.path.exists(image_filename):
        if not download_sample_image(image_url, image_filename):
            print("Örnek resim indirilemedi. Lütfen 'sample_face.jpg' adında bir yüz resmi ekleyin.")
            return

    print(f"\n--- {image_filename} üzerinde Fizyonomi Analizi Başlatılıyor ---\n")
    
    # Run main.py
    cmd = ["python", "main.py", "--source", image_filename]
    subprocess.run(cmd)

    print("\n--- Demo Tamamlandı ---")
    print(f"Sonuçları kontrol edin:\n1. {image_filename} (Girdi)\n2. analysis_result.jpg (Çıktı)\n3. analysis_report.txt (Rapor)")

if __name__ == "__main__":
    main()
