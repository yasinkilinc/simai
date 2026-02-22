# Yapay Zeka ile Yüz Tanıma ve Fizyonomi Analizi Yol Haritası

Bu belge, mevcut uygulamanın yapay zeka yeteneklerini geliştirmek ve daha ileri seviye bir fizyonomi analiz sistemi oluşturmak için izlenecek adımları içerir.

## Faz 1: Veri Toplama ve Hazırlık (1-2 Ay)

Yapay zeka modellerinin başarısı, kaliteli ve etiketli veriye bağlıdır.

### 1.1. Veri Seti Oluşturma
- **Hedef:** 10.000+ yüz fotoğrafından oluşan çeşitli bir veri seti.
- **Kaynaklar:**
    - Açık kaynak veri setleri (FFHQ, CelebA, FairFace).
    - Gönüllü kullanıcı verileri (KVKK/GDPR uyumlu).
    - Tarihsel/Arşiv fotoğrafları (ünlü liderler, düşünürler).
- **Çeşitlilik:** Farklı yaş, cinsiyet, etnik köken ve ışık koşulları.

### 1.2. Etiketleme (Annotation)
- **Fizyonomik Etiketler:** Uzmanlar tarafından belirlenen özellikler.
    - Örn: "Geniş alın", "Kemerli burun", "Ayrık gözler".
- **Kişilik Etiketleri:** Psikolojik test sonuçları (Big Five, MBTI) ile eşleştirilmiş yüz verileri.
- **Etiketleme Araçları:**
    
    #### Seçenek 1: CVAT (Computer Vision Annotation Tool)
    - **Açıklama:** Intel tarafından geliştirilen açık kaynaklı, web tabanlı görüntü ve video etiketleme platformu.
    - **Avantajlar:**
        - Çok kullanıcılı ve işbirliğine dayalı çalışma imkanı.
        - Polygon, polyline, point, bounding box gibi zengin etiketleme araçları.
        - Video etiketleme desteği (frame-by-frame).
        - REST API ile otomasyon desteği.
        - Kalite kontrol ve görev yönetimi özellikleri.
        - Etiket interpolasyonu (video için otomatik ara frame etiketleme).
    - **Dezavantajlar:**
        - Sunucu kurulumu ve bakımı gerektirir.
        - İlk kurulum karmaşık olabilir.
    - **Kullanım Senaryosu:** Ekip çalışması, büyük ölçekli veri seti etiketleme, profesyonel kullanım.
    
    #### Seçenek 2: LabelImg
    - **Açıklama:** Hafif ve masaüstü tabanlı görüntü etiketleme aracı.
    - **Avantajlar:**
        - Basit ve kullanımı kolay arayüz.
        - Kurulum gerektirmez (standalone exe).
        - Hızlı başlangıç için ideal.
        - Pascal VOC ve YOLO formatlarında çıktı.
    - **Dezavantajlar:**
        - Sınırlı etiketleme türleri (sadece bounding box).
        - Çok kullanıcılı çalışma desteği yok.
        - Video etiketleme özelliği yok.
        - Landmark/keypoint etiketleme için uygun değil.
    - **Kullanım Senaryosu:** Küçük ölçekli projeler, hızlı prototipleme, tek kullanıcı.
    
    #### Seçenek 3: Özel Etiketleme Aracı (Tavsiye Edilen)
    - **Açıklama:** Fizyonomi analizi için özel geliştirilmiş PyQt6 tabanlı masaüstü etiketleme uygulaması.
    - **Neden Geliştirilmeli:**
        - Mevcut AnnotationView altyapısı zaten var.
        - Fizyonomi etiketleme için özelleşmiş özellikler:
            - MediaPipe ile otomatik ön-etiketleme.
            - 468 yüz landmark'ının manuel düzenlemesi.
            - Fizyonomik bölge grupları (alın, göz, burun, çene vb.) için özel UI.
            - Batch işleme ve otomatik kaydetme.
        - Kullanıcı deneyimi tam kontrolünüz altında.
        - Veri formatı ve veritabanı ile tam entegrasyon.
    - **Geliştirme Adımları:**
        - Mevcut `AnnotationView` üzerinden standalone mod.
        - Bulk import/export özelliği.
        - Klavye kısayolları ile hızlı etiketleme.
        - İlerleme takibi ve kalite kontrol paneli.
        - Her etiketçi için kullanıcı bazlı istatistikler.
    - **Tahmini Geliştirme Süresi:** 1-2 hafta.
    
    #### Karşılaştırma ve Öneri
    
    | Özellik | CVAT | LabelImg | Özel Araç |
    |---------|------|----------|-----------|
    | Landmark Etiketleme | ✅ Var | ❌ Yok | ✅ Optimize |
    | Otomatik Ön-Etiketleme | ⚠️ Model entegrasyonu gerekir | ❌ Yok | ✅ MediaPipe entegre |
    | Kullanım Kolaylığı | ⚠️ Orta | ✅ Çok Kolay | ✅ Projeye Özel |
    | Çok Kullanıcılı | ✅ Var | ❌ Yok | ⚠️ Eklenebilir |
    | Maliyet | 🆓 Ücretsiz | 🆓 Ücretsiz | 💰 Geliştirme zamanı |
    
    **Final Öneri:** İlk aşamada **LabelImg** veya mevcut **AnnotationView** ile küçük bir pilot veri seti (~100 görüntü) etiketleyip etiketleme iş akışını test edin. Ardından daha büyük ölçekli etiketleme için **Özel Etiketleme Aracı**nı geliştirin veya ekip çalışması gerekiyorsa **CVAT** kurulumunu yapın.

## Faz 2: Model Geliştirme ve Eğitimi (2-3 Ay)

Mevcut kural tabanlı (geometrik) sistemden, derin öğrenme tabanlı sisteme geçiş.

### 2.1. Yüz Özellik Çıkarımı (Feature Extraction)

#### Mevcut Sistem vs Hedef Sistem

| Özellik | Mevcut (MediaPipe) | Hedef (Deep Learning) |
|---------|-------------------|----------------------|
| Yöntem | Geometrik landmark'lar (468 nokta) | CNN/Transformer embeddings |
| Özellik Türü | Koordinatlar ve mesafeler | Yüksek seviye soyut özellikler |
| Boyut | ~1400 değer (468×3) | 512-2048 boyutlu vektör |
| Avantajlar | Hızlı, yorumlanabilir | Daha güçlü genelleme, doku/renk bilgisi |
| Dezavantajlar | Sınırlı özellik çeşitliliği | Daha fazla hesaplama gücü gerektirir |

#### Önerilen Model Mimarileri

##### Seçenek 1: ResNet50 (Basit ve Etkili)
- **Açıklama:** Microsoft tarafından geliştirilen 50 katmanlı residual network.
- **Çıktı:** 2048 boyutlu embedding vektörü.
- **Transfer Learning:** ImageNet ön-eğitimli model kullanılabilir.
- **Avantajlar:**
    - Kanıtlanmış başarı oranı.
    - PyTorch/TensorFlow'da hazır model var.
    - Orta seviye hesaplama gücü yeterli.
- **Kullanım:**
    ```python
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    
    # Ön-eğitimli ResNet50 yükle
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Son katmanı kaldır, embedding al
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Yüz fotoğrafından embedding çıkar
    with torch.no_grad():
        embedding = model(face_tensor)  # [1, 2048]
    ```

##### Seçenek 2: EfficientNet-B4 (Daha Hafif ve Hızlı)
- **Açıklama:** Google tarafından geliştirilen verimli model mimarisi.
- **Çıktı:** 1792 boyutlu embedding.
- **Avantajlar:**
    - ResNet50'den 5x daha az parametre.
    - Daha hızlı inference.
    - Mobil/masaüstü uygulamalar için ideal.
- **Kullanım Senaryosu:** Gerçek zamanlı analiz gerekiyorsa.

##### Seçenek 3: ArcFace / MagFace (Yüz Tanıma İçin Optimize)
- **Açıklama:** Yüz tanıma için özel geliştirilmiş loss fonksiyonları ve modeller.
- **Çıktı:** 512 boyutlu normalleştirilmiş embedding.
- **Avantajlar:**
    - Aynı kişinin farklı pozlarını yakın vektörler olarak temsil eder.
    - Farklı kişileri maksimum ayrıştırma.
    - InsightFace kütüphanesi ile kullanıma hazır.
- **Kullanım:**
    ```python
    from insightface.app import FaceAnalysis
    
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    faces = app.get(img)
    if faces:
        embedding = faces[0].embedding  # [512,]
    ```
- **Uyarı:** Fizyonomi için kimlik bilgisi yerine yapısal özellikler önemli, bu yüzden ArcFace + ek özellik çıkarımı kombinasyonu önerilir.

##### Seçenek 4: Vision Transformer (ViT) - En Gelişmiş
- **Açıklama:** Transformer mimarisi görüntü işlemeye uyarlanmış hali.
- **Çıktı:** 768-1024 boyutlu embedding.
- **Avantajlar:**
    - En yüksek doğruluk potansiyeli.
    - Global ilişkileri daha iyi yakalar.
    - Pretrained modeller (Google ViT, DeiT).
- **Dezavantajlar:**
    - Daha fazla veri ve hesaplama gücü gerektirir.
    - Daha yavaş inference.
- **Öneri:** Eğer yeterli veri varsa (~50K+ etiketli görüntü) tercih edilebilir.

#### Hibrit Yaklaşım (Önerilen)
```
MediaPipe Landmarks (Geometrik) + Deep Learning Embeddings (Semantik)
         ↓                                    ↓
    [1400 değer]                         [512 değer]
         ↓                                    ↓
         └────────────── Birleştir ──────────────┘
                            ↓
                   [1912 boyutlu özellik vektörü]
                            ↓
                  Downstream Task (Kişilik Tahmini)
```

**Avantajlar:**
- MediaPipe'ın hassasiyeti + Deep Learning'in genelleme gücü.
- Geometrik anormallikler + görsel doku birlikte değerlendirilebilir.

---


### 2.2. Çoklu Görev Öğrenimi (Multi-Task Learning)

#### Neden Çoklu Görev Öğrenimi?
Tek bir modelin aynı anda birden fazla görevi öğrenmesi:
- **Verimlilik:** Tek model, birden fazla iş gördüğü için kaynak tasarrufu.
- **Genelleme:** Görevler birbirini destekler (örn: yaş tahmini, yüz şekli tespitine yardımcı olur).
- **Özellik Paylaşımı:** Alt katmanlar ortak özellikler öğrenir.

#### Model Mimarisi

```
                    Input: Yüz Görüntüsü (224×224×3)
                              ↓
                    Backbone (ResNet50 / EfficientNet)
                    Shared Feature Extractor
                              ↓
                    Feature Vector [2048-dim]
                              ↓
          ┌───────────────────┼───────────────────┐
          ↓                   ↓                   ↓
    Task Head 1         Task Head 2         Task Head 3
    Yüz Şekli          Yaş & Cinsiyet      Mikro İfadeler
    Sınıflandırma       Regresyon          Sınıflandırma
          ↓                   ↓                   ↓
    [Oval, Kare,        [Yaş: 0-100,        [Mutlu, Ciddi,
     Üçgen, vb.]         Cinsiyet: M/F]      Sinirgial, vb.]
```

#### Task Detayları

##### Task 1: Yüz Şekli Sınıflandırması
- **Kategoriler:** Oval, Yuvarlak, Kare, Üçgen, Uzun, Elmas (6 sınıf).
- **Loss:** CrossEntropyLoss
- **Metrik:** Accuracy, F1-Score
- **Çıktı:** Softmax olasılıkları

##### Task 2: Yaş ve Cinsiyet Tahmini
- **Yaş:** Regresyon görevi (0-100 arası).
    - Loss: MSE (Mean Squared Error) veya MAE (Mean Absolute Error)
    - Metrik: MAE (ortalama ±5 yaş doğruluğu hedefi)
- **Cinsiyet:** Binary sınıflandırma (Erkek/Kadın).
    - Loss: Binary CrossEntropy
    - Metrik: Accuracy

##### Task 3: Mikro İfade Analizi
- **Kategoriler:** Nötr, Mutlu, Üzgün, Ciddi, Odaklanmış, Sinirli (6 sınıf).
- **Loss:** CrossEntropyLoss
- **Metrik:** Accuracy, Confusion Matrix
- **Önemli Not:** Bu görev, çekilen fotoğrafın anındaki ifadeyi değil, kişinin genel yüz yapısından kaynaklanan doğal ifade eğilimini öğrenmelidir.

#### Toplam Loss Fonksiyonu
```python
total_loss = (
    α₁ * loss_face_shape +      # α₁ = 0.3
    α₂ * loss_age +              # α₂ = 0.2
    α₃ * loss_gender +           # α₃ = 0.2
    α₄ * loss_expression         # α₄ = 0.3
)
```
**Ağırlıklar (α):** Her görevin önemine göre ayarlanır (hiperparametre).

#### PyTorch Implementasyonu (Örnek)
```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class MultiTaskFaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone
        backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Task heads
        self.head_shape = nn.Linear(2048, 6)      # 6 yüz şekli
        self.head_age = nn.Linear(2048, 1)         # Yaş regresyonu
        self.head_gender = nn.Linear(2048, 2)      # Cinsiyet
        self.head_expression = nn.Linear(2048, 6)  # 6 ifade
    
    def forward(self, x):
        features = self.features(x).flatten(1)
        return {
            'shape': self.head_shape(features),
            'age': self.head_age(features),
            'gender': self.head_gender(features),
            'expression': self.head_expression(features)
        }
```

#### Eğitim Stratejisi
1. **Başlangıç:** Backbone'u dondur (freeze), sadece task head'leri eğit (5 epoch).
2. **Fine-tuning:** Tüm modeli düşük öğrenme hızıyla eğit (20 epoch).
3. **Learning Rate:** 1e-4 (AdamW optimizer).
4. **Batch Size:** 32-64 (GPU belleğine göre).
5. **Data Augmentation:**
    - Random horizontal flip
    - Color jitter (±10% brightness/contrast)
    - Random rotation (±15°)

---

### 2.3. Fizyonomi Modeli (PhysiognomyNet)

#### Genel Bakış
Bu model, **yüz özelliklerinden kişilik puanlarını tahmin etme** gibi kritik ve hassas bir görevi üstlenir.

> [!WARNING]
> **Etik Uyarı:** Fizyonomi biliminin geçerliliği tartışmalıdır. Bu model, eğlence/kişisel gelişim amaçlı tasarlanmalı ve karar verme süreçlerinde (işe alım, kredi onayı vb.) kullanılmamalıdır.

#### Model Mimarisi

```
Input: Yüz Embedding [512-dim] (ArcFace'den)
      + Geometrik Özellikler [1400-dim] (MediaPipe'dan)
            ↓
      Concatenate → [1912-dim]
            ↓
      Dense Layer (1024) + ReLU + Dropout(0.3)
            ↓
      Dense Layer (512) + ReLU + Dropout(0.3)
            ↓
      Dense Layer (256) + ReLU
            ↓
   ┌────────┼────────┼────────┐
   ↓        ↓        ↓        ↓
 Zeka    Duygu    İrade   Sosyal
  Head     Head     Head     Head
   ↓        ↓        ↓        ↓
 [0-100] [0-100] [0-100]  [0-100]
```

#### Çıktı Boyutları (Önerilen)
Her kişilik boyutu için ayrı bir tahmin:

1. **Zeka Seviyesi** (0-100): Analitik düşünme potansiyeli.
2. **Duygusal Yoğunluk** (0-100): Empatik/duygusal tepki eğilimi.
3. **İrade Gücü** (0-100): Kararlılık ve dayanıklılık.
4. **Sosyal Açıklık** (0-100): Dışadönüklük ve sosyalleşme eğilimi.

> [!TIP]
> Bu boyutlar Big Five kişilik modeline (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) eşlenebilir.

#### Eğitim Veri Kaynakları

##### Seçenek 1: Psikolojik Test Eşleştirmesi (İdeal)
- **Yöntem:** Gönüllülerden yüz fotoğrafı + psikolojik test sonuçları toplamak.
- **Testler:**
    - Big Five Personality Test (OCEAN modeli)
    - MBTI (Myers-Briggs Type Indicator)
    - IQ testleri (Raven's Progressive Matrices vb.)
- **Veri Miktarı:** Minimum 5000 eşleştirilmiş örnek.
- **Zorluk:** Veri toplama maliyetli ve uzun sürebilir.

##### Seçenek 2: Self-Assessment Etiketleme
- **Yöntem:** Kullanıcıların kendi kişilik özelliklerini 0-100 skalasında değerlendirmeleri.
- **Avantajlar:** Hızlı veri toplama.
- **Dezavantajlar:** Subjektif ve güvenilir olmayabilir.

##### Seçenek 3: Uzman Etiketçiler
- **Yöntem:** Psikologlar veya fizyonomi uzmanlarının fotoğrafları değerlendirmesi.
- **Avantajlar:** Daha tutarlı etiketler.
- **Dezavantajlar:** Pahalı ve yavaş.

#### Loss Fonksiyonu
```python
# Regresyon görevi (0-100 arası tahmin)
loss = nn.MSELoss()

# Veya daha robust:
loss = nn.SmoothL1Loss()  # Huber Loss - outlier'lara daha az duyarlı
```

#### Eğitim Süreci

1. **Veri Hazırlığı:**
    - Normalize et: Her kişilik boyutunu [0, 1] aralığına getir.
    - Train/Val/Test split: 70% / 15% / 15%
    - K-Fold Cross Validation (k=5) güvenilirlik için.

2. **Model Eğitimi:**
    ```python
    # Hiperparametreler
    epochs = 50
    batch_size = 64
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ```

3. **Regularization:**
    - Dropout: 0.3-0.4
    - L2 Weight Decay: 1e-5
    - Early Stopping: Validation loss 5 epoch artmazsa dur.

4. **Metrikler:**
    - **MAE (Mean Absolute Error):** Ortalama hata (hedef: <10 puan).
    - **Pearson Korelasyon:** Gerçek vs tahmin korelasyonu (hedef: >0.6).
    - **R² Score:** Açıklanan varyans (hedef: >0.4).

#### Model Validasyonu ve A/B Testi
- **Kullanıcı Geri Bildirimi:** "Bu analiz size ne kadar uyuyor? (1-5)"
- **Uzman Değerlendirmesi:** Psikologların sonuçları değerlendirmesi.
- **Karşılaştırma:** Mevcut geometrik sistem vs PhysiognomyNet karşılaştırması.

#### Örnek Kod: PhysiognomyNet
```python
import torch.nn as nn

class PhysiognomyNet(nn.Module):
    def __init__(self, input_dim=1912, hidden_dims=[1024, 512, 256], num_traits=4):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Her kişilik boyutu için ayrı head
        self.trait_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], 1) for _ in range(num_traits)
        ])
    
    def forward(self, x):
        features = self.shared_layers(x)
        # Çıktılar [0, 100] arasında olmalı
        traits = [torch.sigmoid(head(features)) * 100 for head in self.trait_heads]
        return torch.cat(traits, dim=1)  # [batch_size, 4]
```

#### Beklenen Sonuçlar
- **İlk Versiyon (Baseline):** MAE ~15-20 puan, R² ~0.3
- **Optimized Model:** MAE ~8-12 puan, R² ~0.5-0.6
- **Gerçekçi Beklenti:** Tam doğruluk mümkün değil, ancak genel eğilimleri yakalamak mümkün.

---

### 2.4. Eğitim Altyapısı ve Araçlar

#### Gerekli Donanım
- **GPU:** NVIDIA RTX 3060 (12GB) veya üzeri (önerilir: RTX 4090).
- **RAM:** Minimum 16GB, önerilir 32GB.
- **Depolama:** 500GB SSD (veri setleri ve checkpoint'ler için).

#### Yazılım Yığını
```
Python 3.10+
├── PyTorch 2.0+ (CUDA 11.8)
├── torchvision
├── tensorboard (eğitim takibi)
├── wandb (W&B - opsiyonel, bulut takip)
├── opencv-python
├── albumentations (data augmentation)
├── scikit-learn (metrikler)
└── onnx / onnxruntime (model export)
```

#### Eğitim İzleme (Monitoring)
```python
# TensorBoard ile kayıt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/physiognomy_exp1')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('MAE/validation', val_mae, epoch)
```

#### Checkpoint Yönetimi
```python
# En iyi modeli kaydet
if val_loss < best_val_loss:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, 'checkpoints/best_model.pth')
```

#### Tahmini Eğitim Süreleri
| Model | Veri Seti | GPU | Epoch | Toplam Süre |
|-------|-----------|-----|-------|-------------|
| ResNet50 | 10K görüntü | RTX 3060 | 30 | ~3 saat |
| EfficientNet-B4 | 10K görüntü | RTX 3060 | 30 | ~2 saat |
| Multi-Task Model | 10K görüntü | RTX 3060 | 50 | ~5 saat |
| PhysiognomyNet | 5K+labels | RTX 3060 | 50 | ~2 saat |

## Faz 3: Entegrasyon ve Uygulama (1-2 Ay)

Eğitilen modellerin masaüstü uygulamasına entegrasyonu.

### 3.1. Model Optimizasyonu
- **ONNX Runtime:** Modellerin farklı donanımlarda hızlı çalışması için ONNX formatına dönüştürülmesi.
- **Quantization:** Model boyutunu küçültmek ve hızı artırmak (FP32 -> INT8).

### 3.2. Hibrit Sistem
- **Kural Tabanlı + AI:**
    - MediaPipe ile hassas ölçümler (milimetrik analiz).
    - Deep Learning ile genel izlenim ve doku analizi (kırışıklıklar, cilt kalitesi).
- İki sistemin sonuçlarının ağırlıklı ortalaması ile nihai rapor oluşturma.

### 3.3. Geri Bildirim Döngüsü (Active Learning)
- Kullanıcıların analiz sonuçlarına verdiği geri bildirimlerin ("Bu analiz bana uyuyor/uymuyor") toplanması.
- Bu verilerin modelleri yeniden eğitmek (Fine-tuning) için kullanılması.

## Faz 4: İleri Seviye Özellikler (Gelecek Vizyonu)

### 4.1. 3D Yüz Rekonstrüksiyonu (Gelişmiş)
- Tek bir fotoğraftan fotogerçekçi 3D kafa modeli oluşturma (DECA, 3DDFA_V2).
- Yan profil analizinin 3D model üzerinden otomatik yapılması.

### 4.2. Zaman İçinde Değişim Analizi
- Kullanıcının eski ve yeni fotoğraflarını karşılaştırarak yüzdeki ve karakterdeki değişimlerin analizi.
- "Yaşlandırma" simülasyonu ve gelecekteki potansiyel değişimler.

### 4.3. Video Analizi
- Anlık video akışı üzerinden jest, mimik ve mikro ifade analizi.
- Konuşma sırasındaki tutum ve davranış analizi.

## Teknoloji Yığını (Öneri)

- **Dil:** Python
- **Framework:** PyTorch (Eğitim), ONNX Runtime (Inference)
- **Görüntü İşleme:** OpenCV, Pillow
- **Yüz Kütüphaneleri:** MediaPipe, InsightFace, dlib
- **Veri Tabanı:** PostgreSQL (Vektör verileri için pgvector eklentisi)
