# Veri Seti ve Etiketleme Stratejisi

Bu belge, yüz analizi ve fizyonomi projesi için kapsamlı bir veri seti oluşturma ve etiketleme stratejisini detaylandırır.

## 1. Veri Seti Oluşturma Stratejisi (Faz 1.1)

Hedefimiz, yapay zeka modellerinin genellenebilirliğini artırmak için çeşitli, dengeli ve yüksek kaliteli bir veri seti oluşturmaktır.

### 1.1.1. Veri Kaynakları ve Toplama Yöntemleri

| Kaynak Tipi | Açıklama | Hedef Sayı | Aksiyon |
| :--- | :--- | :--- | :--- |
| **Açık Kaynak Veri Setleri** | FFHQ (Flickr-Faces-HQ), CelebA-HQ, FairFace gibi akademik veri setleri. Yüksek çözünürlük ve çeşitlilik sağlar. | 5.000+ | İlgili veri setlerini indir ve proje formatına uygun hale getir. |
| **Tarihsel Arşiv** | Liderler, filozoflar, sanatçılar gibi karakter analizi yapılmış ünlü kişilerin fotoğrafları. Fizyonomi literatürü ile eşleştirme için kritik. | 500+ | Web scraping ve arşiv taraması ile topla. |
| **Gönüllü Kullanıcı Verileri** | Uygulamayı kullanan ve veri paylaşımına izin veren gerçek kullanıcılar. Gerçek dünya senaryoları için en değerli veri. | Sürekli | Uygulamaya "Veri Paylaşımı" onayı ekle. |
| **Sentetik Veri** | StyleGAN gibi araçlarla üretilen, belirli özellikleri (örn: çok geniş alın) abartılmış yüzler. Nadir özellikleri eğitmek için. | 1.000+ | StyleGAN3 kullanarak spesifik özelliklere sahip yüzler üret. |

### 1.1.2. Veri Standartları ve Önişleme

Toplanan her veri aşağıdaki standartlara getirilmelidir:

1.  **Format:** JPG veya PNG.
2.  **Çözünürlük:** Minimum 1024x1024 piksel.
3.  **Hizalama:** Yüz ortalanmış, gözler yatay düzlemde hizalı.
4.  **Temizlik:** Bulanık, aşırı karanlık veya yüzün %20'sinden fazlası kapalı olan fotoğraflar elenmeli.
5.  **Klasör Yapısı:**
    ```
    dataset/
      ├── raw/            # Ham indirilmiş veriler
      ├── processed/      # İşlenmiş, hizalanmış veriler
      │   ├── train/
      │   ├── val/
      │   └── test/
      └── metadata.csv    # Her fotoğraf için ID, kaynak, izin durumu vb.
    ```

---

## 2. Etiketleme (Annotation) Stratejisi (Faz 1.2)

Veri setinin değerini belirleyen en önemli faktör etiketlerin kalitesidir. İki tür etiketleme yapılacaktır: Fizyonomik (Fiziksel) ve Psikolojik (Karakter).

### 2.1. Fizyonomik Etiketleme (Objektif)

Yüzün fiziksel özelliklerinin ölçülmesi ve sınıflandırılmasıdır.

**Etiket Kategorileri:**

*   **Yüz Şekli:** Oval, Kare, Yuvarlak, Kalp, Dikdörtgen, Elmas, Üçgen.
*   **Alın:** Genişlik (Dar/Geniş), Yükseklik (Kısa/Yüksek), Eğim (Düz/Eğimli).
*   **Gözler:** Büyüklük, Eğim (Çekik/Düşük), Aralık (Ayrık/Bitişik), Çukur/Çıkık.
*   **Burun:** Uzunluk, Genişlik, Kemer Durumu, Burun Ucu Şekli.
*   **Dudaklar:** Kalınlık (Alt/Üst), Genişlik.
*   **Çene:** Genişlik, Çıkıklık, Gamze Durumu.
*   **Kulaklar:** Büyüklük, Kepçelik Durumu, Lob Yapısı.

**Yöntem:**
1.  **Otomatik Ön Etiketleme:** MediaPipe kullanarak geometrik ölçümlerle (oranlar) taslak etiketler oluşturulur.
2.  **İnsan Doğrulaması:** Uzmanlar veya eğitilmiş etiketleyiciler, otomatik etiketleri kontrol eder ve düzeltir (Human-in-the-loop).

### 2.2. Psikolojik Etiketleme (Subjektif/Korelasyonel)

Fizyonomi literatürüne dayalı karakter özelliklerinin atanmasıdır.

**Etiket Kategorileri (Big Five + Fizyonomi):**

*   **Zeka/Analitik Yetenek:** (Alın yapısı ile ilişkili)
*   **Duygusallık/Empati:** (Gözler ve dudaklar ile ilişkili)
*   **İrade/Kararlılık:** (Çene ve burun yapısı ile ilişkili)
*   **Enerji/Dürtüsellik:** (Yüz genişliği ve kaşlar ile ilişkili)
*   **Sosyal Yetenekler:** (Göz çevresi ve ağız yapısı ile ilişkili)

**Yöntem:**
1.  **Literatür Eşleşmesi:** Tanımlanmış fizyonomik özelliklere karşılık gelen geleneksel fizyonomi yorumları otomatik atanır.
2.  **Psikometrik Test Eşleşmesi:** Gönüllü kullanıcılardan alınan MBTI veya Big Five test sonuçları, doğrudan yüz verisiyle eşleştirilir (Ground Truth).

### 2.3. Etiketleme Araçları ve İş Akışı

1.  **CVAT (Computer Vision Annotation Tool):**
    *   Yüz bounding box ve landmark düzeltmeleri için.
    *   Web tabanlı, çoklu kullanıcı desteği.
2.  **Özel Etiketleme Arayüzü (Streamlit/PyQt):**
    *   Hızlı sınıflandırma için (Örn: Ekrana yüz gelir, sağda "Geniş Alın", "Dar Alın" butonları olur).
    *   Fizyonomik özelliklerin hızlıca işaretlenmesi için optimize edilmiş arayüz.

---

## 3. Görev Listesi (Task List)

### Hazırlık
- [ ] **Veri Toplama Scriptleri:**
    - [ ] FFHQ ve FairFace veri setlerini indiren scriptler yaz.
    - [ ] Google Images / Bing Images için "famous portraits" araması yapan scraper yaz.
- [ ] **Veri Temizleme Pipeline'ı:**
    - [ ] Yüz tespiti yapamayan fotoğrafları eleyen script.
    - [ ] Çözünürlüğü düşük olanları eleyen script.
    - [ ] Yüz hizalama (alignment) yapan script (gözleri yatay düzleme getirme).

### Etiketleme Altyapısı
- [ ] **Etiketleme Şeması (Schema) Tanımlama:**
    - [ ] JSON formatında tüm fizyonomik ve karakter etiketlerinin yapısını oluştur.
- [ ] **Otomatik Etiketleme Aracı:**
    - [ ] Mevcut `AnalysisEngine` sınıfını kullanarak toplu (batch) analiz yapan ve sonuçları JSON/CSV olarak kaydeden bir araç geliştir.
- [ ] **Manuel Doğrulama Arayüzü:**
    - [ ] Streamlit kullanarak basit bir arayüz yap:
        - Sol: Yüz Fotoğrafı + Heatmap.
        - Sağ: Modelin tahmin ettiği etiketler (Dropdown/Checkbox).
        - Aksiyon: "Onayla" veya "Düzelt ve Kaydet".

### Uygulama
- [ ] **Pilot Veri Seti (İlk 1000):**
    - [ ] 1000 fotoğraflık dengeli bir alt küme oluştur.
    - [ ] Otomatik etiketle ve manuel doğrula.
    - [ ] Model eğitiminde baseline olarak kullan.
