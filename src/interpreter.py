import json
import os
from collections import Counter

from src.features import FaceFeatures

class RuleEngine:
    def __init__(self, rules_path='src/rules/ilmi_sima_etiket_kutuphanesi.json'):
        self.rules_path = rules_path
        self.rules = self._load_rules()

    def _load_rules(self):
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('fizyonomi_master_veritabani', {})
        except Exception as e:
            print(f"Error loading rules from {self.rules_path}: {e}")
            return {}

    def apply_rules(self, features: FaceFeatures):
        report = {
            'face_shape': features.face_shape,
            'analysis': {
                'positive': [],
                'negative': [],
                'neutral': []
            },
            'cross_validated_traits': [],
            'annotations': features.annotations
        }
        
        # Map-Reduce için etiket havuzu
        tag_counts = Counter()

        def add_finding(rules_category, variation_name, report_category):
            category_rules = self.rules.get(rules_category, [])
            for r in category_rules:
                if r.get('varyasyon') == variation_name:
                    item = {
                        "trait": variation_name,
                        "description": r.get('anlam', '')
                    }
                    report['analysis'][report_category].append(item)
                    # Etiketleri (tags) havuza ekle
                    for tag in r.get('etiketler', []):
                        tag_counts[tag] += 1
                    return True
            return False

        # Yüz Şekli Fallback
        report['analysis']['neutral'].append({
            "organ": "Yüz Şekli",
            "trait": features.face_shape,
            "description": f"Yüz şekliniz {features.face_shape} olarak tespit edildi."
        })

        # 1. Alın Yapısı
        alin_info = features.annotations.get('forehead', {})
        f_width = alin_info.get('width', 'Normal')
        f_height = alin_info.get('height', 'Normal')
        
        if f_width == 'Geniş' or f_height == 'Yüksek':
            add_finding('alin_yapisi', "Geniş Alın (Yatayda Uzun)", 'positive')
        elif f_width == 'Dar' or f_height == 'Kısa':
            add_finding('alin_yapisi', "Dar Alın (Yatayda Kısa)", 'negative')

        # 2. Kaş Yapısı
        e_thickness = features.metrics.get('eyebrow_thickness', 10)
        if e_thickness < 5:
            add_finding('kas_yapisi', "İnce ve Seyrek Kaş", 'neutral') # Kurnaz, Zihinsel

        # 3. Göz Yapısı
        eyes_info = features.annotations.get('eyes', {})
        e_size = eyes_info.get('size', 'Normal')
        e_slant = eyes_info.get('slant', 'Düz')

        if e_size == 'Küçük':
            add_finding('goz_yapisi', "Kısık ve İnce Gözler (Avcı)", 'negative') # Kurnaz
        if e_slant == 'Düşük':
            add_finding('goz_yapisi', "Dış Kantus Aşağı Eğik", 'negative')
        
        # 4. Burun Yapısı
        nose_info = features.annotations.get('nose', {})
        n_width = nose_info.get('width', 'Normal')
        n_length = nose_info.get('length', 'Normal')
        n_shape = nose_info.get('shape', 'Normal')
        n_tip = nose_info.get('tip', 'Normal')
        
        if n_shape == 'Gaga' or n_tip == 'Düşük':
            add_finding('burun_yapisi', "Gaga Burun (Ucu Aşağı Düşük)", 'negative') # Kurnaz
        elif n_width == 'Geniş':
             add_finding('burun_yapisi', "Geniş Burun Delikleri", 'negative')
        elif n_length == 'Uzun':
             add_finding('burun_yapisi', "Büyük ve Etli Burun", 'neutral')

        # 5. Dudaklar ve Ağız
        lips_info = features.annotations.get('lips', {})
        l_upper = lips_info.get('upper_thickness', 'Normal')
        l_lower = lips_info.get('lower_thickness', 'Normal')
        
        if l_upper == 'İnce':
            add_finding('filtrum_ve_dudaklar', "İnce Üst Dudak", 'neutral')
        if l_lower == 'Kalın':
            add_finding('filtrum_ve_dudaklar', "Kalın Alt Dudak", 'positive')

        # 6. Çene
        chin_info = features.annotations.get('chin', {})
        c_width = chin_info.get('width', 'Normal')

        if c_width == 'Geniş':
             add_finding('cene_ve_yanaklar', "Geniş ve Köşeli Çene", 'positive')
        elif c_width == 'Dar':
             add_finding('cene_ve_yanaklar', "Sivri Çene", 'neutral')

        # --- Çapraz Doğrulama Hesaplaması (Map-Reduce) ---
        bilişsel_tags = {"Zeki", "Entelektüel", "Analitik", "Detaycı", "Mükemmeliyetçi", "Hafızası_Güçlü", "Odak_Problemi", "Sabit_Fikirli", "Dar_Görüşlü", "Pratik", "Gözlemci"}
        sosyal_tags = {"İletişimi_Güçlü", "Diplomatik", "Dışa_Dönük", "İçten_Pazarlıklı", "Geveze", "Sivri_Dilli", "Açık_Sözlü", "Ketum", "Sır_Küpü", "Manipülatif", "Yalancı", "Güvenilmez", "Kibirli", "Lider", "Narsist", "Uyumlu", "Bireysel"}
        duygu_tags = {"Duygusal", "Enerjik", "Hassas", "Kırılgan", "Evhamlı", "Güçlü", "İradeli", "İnatçı", "Mücadeleci", "Hırslı", "Sabırsız", "Sabırlı", "Soğukkanlı", "Cesur", "Korkak", "Gururlu", "Bencil", "Cömert", "Pasif", "Kindar", "Öfkeli", "Agresif"}

        detailed_analysis = {
            "Zihin ve Bilişsel Yapı": [],
            "Sosyal ve İletişim": [],
            "Duygu ve İrade": [],
            "Genel Karakter": []
        }

        for tag, count in tag_counts.items():
            trait_name = tag.replace("_", " ")
            if count >= 3:
                report['cross_validated_traits'].append({
                    "trait": trait_name,
                    "count": count
                })
            
            # Detaylı analiz için tek geçişte bile olanları toplayalım ama vurguyu farklı yapabiliriz
            # Şimdilik yüzdeki tüm tespit edilen etiketleri listele
            if count >= 1:
                if tag in bilişsel_tags:
                    detailed_analysis["Zihin ve Bilişsel Yapı"].append(trait_name)
                elif tag in sosyal_tags:
                    detailed_analysis["Sosyal ve İletişim"].append(trait_name)
                elif tag in duygu_tags:
                    detailed_analysis["Duygu ve İrade"].append(trait_name)
                else:
                    detailed_analysis["Genel Karakter"].append(trait_name)
                    
        # Temizleme (Boş listeleri çıkart)
        report['detailed_analysis'] = {k: list(set(v)) for k, v in detailed_analysis.items() if v}

        return report


class PhysiognomyInterpreter:
    def __init__(self):
        self.engine = RuleEngine()

    def interpret(self, features: FaceFeatures):
        """
        Features nesnesini alır ve JSON kurallarını (RuleEngine) kullanarak çapraz doğrulanmış analiz raporu üretir.
        """
        return self.engine.apply_rules(features)
