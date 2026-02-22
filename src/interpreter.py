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

        # =====================================================
        # 1. ALIN YAPISI
        # =====================================================
        alin_info = features.annotations.get('forehead', {})
        f_width = alin_info.get('width', 'Normal')
        f_height = alin_info.get('height', 'Normal')
        f_slope = alin_info.get('slope', 'Normal')
        
        # Geniş / Dar Alın
        if f_width == 'Geniş':
            add_finding('alin_yapisi', "Geniş Alın (Yatayda Uzun)", 'positive')
        elif f_width == 'Dar':
            add_finding('alin_yapisi', "Dar Alın (Yatayda Kısa)", 'negative')
        
        # Dikey Uzun/Kısa Alın
        if f_height == 'Yüksek':
            add_finding('alin_yapisi', "Dikeyde Uzun Alın", 'positive')
        elif f_height == 'Kısa':
            add_finding('alin_yapisi', "Dikeyde Kısa Alın", 'negative')
            
        # Alın Eğimi (Geriye Yatık / Dik) 
        if f_slope == 'Eğimli':
            add_finding('alin_yapisi', "Geriye Yatık Alın", 'neutral')
        elif f_slope == 'Yuvarlak':
            add_finding('alin_yapisi', "Merhamet Çatısı (Kaş üstü şişkinlik)", 'positive')

        # =====================================================
        # 2. KAŞ YAPISI
        # =====================================================
        eyebrows_info = features.annotations.get('eyebrows', {})
        e_thickness = eyebrows_info.get('thickness', 'Normal')
        
        if e_thickness == 'Kalın' or e_thickness == 'Gür':
            add_finding('kas_yapisi', "Gür ve Kalın Kaş", 'positive')
        elif e_thickness == 'İnce' or e_thickness == 'Seyrek':
            add_finding('kas_yapisi', "İnce ve Seyrek Kaş", 'neutral')
        
        # Kaş-göz mesafesi (metrics'ten)
        eyebrow_arch = features.metrics.get('eyebrow_arch', 0)
        if eyebrow_arch > 0.015:
            add_finding('kas_yapisi', "Gözden Uzak Kaş", 'positive')  # Sabırlı
        elif eyebrow_arch < -0.005:
            add_finding('kas_yapisi', "Göze Yakın Kaş", 'neutral')  # Sabırsız

        # =====================================================
        # 3. GÖZ YAPISI
        # =====================================================
        eyes_info = features.annotations.get('eyes', {})
        e_size = eyes_info.get('size', 'Normal')
        e_slant = eyes_info.get('slant', 'Düz')
        e_depth_cat = eyes_info.get('depth', 'Normal')

        # Göz Büyüklüğü
        if e_size == 'Büyük':
            add_finding('goz_yapisi', "Büyük Gözler", 'positive')
        elif e_size == 'Küçük':
            add_finding('goz_yapisi', "Kısık ve İnce Gözler (Avcı)", 'negative')
        
        # Göz Eğimi
        if e_slant == 'Düşük' or e_slant == 'Aşağı':
            add_finding('goz_yapisi', "Dış Kantus Aşağı Eğik", 'negative')
        
        # Göz Derinliği (Çukur vs Pörtlek)
        eye_depth = features.metrics.get('eye_depth', 0)
        if e_depth_cat == 'Çukur' or eye_depth > 0.02:
            add_finding('goz_yapisi', "Çukur Gözler", 'neutral')
        elif e_depth_cat == 'Çıkık' or eye_depth < -0.01:
            add_finding('goz_yapisi', "Pörtlek (Dışarı Çıkık) Gözler", 'negative')

        # =====================================================
        # 4. BURUN YAPISI
        # =====================================================
        nose_info = features.annotations.get('nose', {})
        n_width = nose_info.get('width', 'Normal')
        n_length = nose_info.get('length', 'Normal')
        n_bridge = nose_info.get('bridge', 'Normal')
        n_tip = nose_info.get('tip', 'Normal')
        
        # Burun Kemeri
        if n_bridge == 'Kemerli':
            add_finding('burun_yapisi', "Kemerli Burun", 'positive')
        
        # Burun Ucu
        if n_tip == 'Kalkık':
            add_finding('burun_yapisi', "Burun Ucu Kalkık", 'positive')  # Cömert
        elif n_tip == 'Düşük':
            add_finding('burun_yapisi', "Gaga Burun (Ucu Aşağı Düşük)", 'negative')
            
        # Burun Genişliği
        if n_width == 'Geniş':
            add_finding('burun_yapisi', "Geniş Burun Delikleri", 'negative')
        elif n_width == 'Dar' or n_width == 'İnce':
            add_finding('burun_yapisi', "İnce Burun", 'neutral')
            
        # Burun Uzunluğu
        if n_length == 'Uzun':
            add_finding('burun_yapisi', "Büyük ve Etli Burun", 'neutral')

        # Burun Ucu Şekli (Yuvarlak = Top Burun)
        nose_tip_angle = features.metrics.get('nose_tip_angle', 90)
        if nose_tip_angle > 105:
            add_finding('burun_yapisi', "Top Burun (Yuvarlak Uçlu)", 'positive')

        # =====================================================
        # 5. DUDAKLAR VE AĞIZ
        # =====================================================
        lips_info = features.annotations.get('lips', {})
        l_upper = lips_info.get('upper_thickness', 'Normal')
        l_lower = lips_info.get('lower_thickness', 'Normal')
        l_width = lips_info.get('width', 'Normal')
        
        # Üst Dudak
        if l_upper == 'Kalın':
            add_finding('filtrum_ve_dudaklar', "Kalın Üst Dudak", 'positive')
        elif l_upper == 'İnce':
            add_finding('filtrum_ve_dudaklar', "İnce Üst Dudak", 'neutral')
        
        # Alt Dudak
        if l_lower == 'Kalın':
            add_finding('filtrum_ve_dudaklar', "Kalın Alt Dudak", 'positive')
        
        # Ağız Kenarı Açısı (Mutlu/Somurtkan yüz)
        mouth_corner_drop = features.metrics.get('mouth_corner_drop', 0)
        if mouth_corner_drop > 0.008:
            # Kenarlar aşağı düşük -> Somurtkan/Kindar ifade
            report['analysis']['neutral'].append({
                "trait": "Aşağı Düşük Ağız Kenarları",
                "description": "Ağız kenarlarının aşağı düşmesi memnuniyetsizlik veya kararlılık işareti olabilir."
            })
            tag_counts["Kindar"] += 1
            tag_counts["Memnuniyetsiz"] += 1
        elif mouth_corner_drop < -0.005:
            # Kenarlar yukarı kalkık -> Neşeli/Cömert ifade
            report['analysis']['positive'].append({
                "trait": "Yukarı Kalkık Ağız Kenarları",
                "description": "Doğuştan gülümseyen ağız yapısı iyimserlik ve cömertlik işareti."
            })
            tag_counts["Cömert"] += 1
            tag_counts["Dışa_Dönük"] += 1

        # =====================================================
        # 6. ÇENE VE YANAKLAR
        # =====================================================
        chin_info = features.annotations.get('chin', {})
        c_width = chin_info.get('width', 'Normal')
        c_prominence = chin_info.get('prominence', 'Normal')
        c_dimple = chin_info.get('dimple', 'Yok')

        # Çene Genişliği
        if c_width == 'Geniş':
            add_finding('cene_ve_yanaklar', "Geniş ve Köşeli Çene", 'positive')
        elif c_width == 'Dar':
            add_finding('cene_ve_yanaklar', "Sivri Çene", 'neutral')
            
        # Çene Çıkıklığı
        if c_prominence == 'Çıkık':
            report['analysis']['positive'].append({
                "trait": "Çıkık Çene",
                "description": "Güçlü irade ve kararlılık göstergesi."
            })
            tag_counts["İradeli"] += 1
            tag_counts["Mücadeleci"] += 1
        elif c_prominence == 'Geride':
            add_finding('cene_ve_yanaklar', "Geride (Çekik) Çene", 'negative')
        
        # Gamze
        if c_dimple == 'Var':
            add_finding('cene_ve_yanaklar', "Gamzeli Çene", 'positive')
        
        # Elmacık Kemiği Belirginliği
        cheek_jaw_ratio = features.metrics.get('cheek_jaw_ratio', 1.0)
        if cheek_jaw_ratio > 1.25:
            add_finding('cene_ve_yanaklar', "Elmacık Kemiği Belirgin", 'positive')
            
        # =====================================================
        # 7. YÜZSEL ASİMETRİ KONTROLÜ
        # =====================================================
        forehead_asym = features.metrics.get('forehead_asymmetry', 0)
        if abs(forehead_asym) > 0.015:
            report['analysis']['neutral'].append({
                "trait": "Yüzsel Asimetri Tespit Edildi",
                "description": "Yüzün sağ ve sol tarafı arasında belirgin bir asimetri mevcut. Bu iç dünyada çatışma veya çift karakterli yapıya işaret edebilir."
            })
            tag_counts["Dengesiz"] += 1
            tag_counts["Tutarsız"] += 1

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
