import cv2
import numpy as np
import json
import os
from datetime import datetime

try:
    import mediapipe as mp
except ImportError:
    mp = None

class AutoAnnotator:
    """Automatic physiognomic feature annotation using MediaPipe"""
    
    def __init__(self):
        self.face_mesh = None
        if mp:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
    
    def annotate_image(self, image_source):
        """Generate automatic annotations for an image (path or numpy array)"""
        if self.face_mesh is None:
            return self._default_annotations()
        
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
        elif isinstance(image_source, np.ndarray):
            img = image_source.copy()
        else:
            return self._default_annotations()
            
        if img is None:
            return self._default_annotations()
        
        h, w = img.shape[:2]
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return self._default_annotations()
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate common metrics
        metrics = self.calculate_zone_metrics(img, landmarks)
        
        # Calculate advanced ratios for ML
        ml_features = self.calculate_ml_features(metrics, landmarks, w, h)
        
        # Calculate features
        annotations = {
            'face_shape': self._detect_face_shape(metrics, landmarks, w, h),
            'forehead': self._analyze_forehead(metrics, landmarks, w, h),
            'eyes': self._analyze_eyes(metrics, landmarks, w, h),
            'nose': self._analyze_nose(metrics, landmarks, w, h),
            'lips': self._analyze_lips(metrics, landmarks, w, h),
            'chin': self._analyze_chin(metrics, landmarks, w, h),
            'ears': self._analyze_ears(landmarks, w, h),
            'metrics': ml_features, # Raw metrics for ML training
            'auto_generated': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return annotations

    def calculate_ml_features(self, metrics, landmarks, w, h):
        """Calculate normalized feature vector for ML models"""
        features = {}
        
        # Handle NormalizedLandmarkList
        if hasattr(landmarks, 'landmark'):
            landmarks = landmarks.landmark
            
        # Basic Dimensions
        face_h = metrics.get('face_height', 1)
        if face_h == 0: face_h = 1
        
        z1_h = metrics.get('z1_h', 0)
        z2_h = metrics.get('z2_h', 0)
        z3_h = metrics.get('z3_h', 0)
        
        z1_w = metrics.get('z1_w', 0)
        z2_w = metrics.get('z2_w', 0)
        z3_w = metrics.get('z3_w', 0)
        
        # 1. Vertical Proportions
        features['z1_h_ratio'] = z1_h / face_h
        features['z2_h_ratio'] = z2_h / face_h
        features['z3_h_ratio'] = z3_h / face_h
        
        # 2. Horizontal Proportions (relative to cheek width z2_w)
        ref_w = z2_w if z2_w > 0 else 1
        features['z1_w_ratio'] = z1_w / ref_w
        features['z3_w_ratio'] = z3_w / ref_w
        features['face_wh_ratio'] = ref_w / face_h
        
        # 3. Eye Metrics
        l_eye_w = abs(landmarks[33].x - landmarks[133].x) * w
        r_eye_w = abs(landmarks[263].x - landmarks[362].x) * w
        avg_eye_w = (l_eye_w + r_eye_w) / 2
        icd = abs(landmarks[133].x - landmarks[362].x) * w # Inter-canthal distance
        
        features['eye_spacing_ratio'] = icd / avg_eye_w if avg_eye_w > 0 else 0
        features['eye_size_ratio'] = avg_eye_w / ref_w
        
        # 4. Nose Metrics
        nose_w = abs(landmarks[102].x - landmarks[331].x) * w
        features['nose_width_ratio'] = nose_w / ref_w
        features['nose_icd_ratio'] = nose_w / icd if icd > 0 else 0
        
        # 5. Lips Metrics
        mouth_w = abs(landmarks[61].x - landmarks[291].x) * w
        features['mouth_width_ratio'] = mouth_w / ref_w
        
        # Lip thickness
        ul_h = abs(landmarks[13].y - landmarks[14].y) * h
        ll_h = abs(landmarks[14].y - landmarks[17].y) * h
        features['upper_lip_ratio'] = ul_h / face_h
        features['lower_lip_ratio'] = ll_h / face_h
        
        # 6. Chin/Jaw Metrics
        jaw_w = abs(landmarks[397].x - landmarks[172].x) * w
        features['jaw_cheek_ratio'] = jaw_w / ref_w
        
        return features

    def get_landmarks(self, image_source):
        """Get raw landmarks for an image"""
        if self.face_mesh is None:
            return None
            
        if isinstance(image_source, str):
            img = cv2.imread(image_source)
        elif isinstance(image_source, np.ndarray):
            img = image_source.copy()
        else:
            return None
            
        if img is None:
            return None
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            return None
            
        return results.multi_face_landmarks[0].landmark

    def detect_hairline_y(self, img, landmarks):
        """Detect hairline Y coordinate using image processing"""
        try:
            h, w = img.shape[:2]
            
            # Handle NormalizedLandmarkList
            if hasattr(landmarks, 'landmark'):
                landmarks = landmarks.landmark
            
            # Check length
            if len(landmarks) < 153:
                print(f"Warning: Landmarks list too short ({len(landmarks)}), cannot detect hairline.")
                return 0 # Or some default
            
            # Key Landmarks
            lm_10 = landmarks[10] # Mesh Top
            lm_brow = landmarks[105] # Brow Center Approx
            lm_nose = landmarks[2] # Nose Bottom
            lm_chin = landmarks[152] # Chin Bottom
            
            # Y Coordinates
            brow_y = int(lm_brow.y * h)
            nose_y = int(lm_nose.y * h)
            chin_y = int(lm_chin.y * h) + int(h * 0.015)
            
            # Calculate Zone 2 and Zone 3 heights
            zone2_h = abs(nose_y - brow_y)
            zone3_h = abs(chin_y - nose_y)
            
            # Expected Zone 1 height (Average of Zone 2 and 3)
            avg_zone_h = (zone2_h + zone3_h) / 2
            
            # Estimated Hairline Y
            estimated_hairline_y = int(brow_y - avg_zone_h)
            
            # Define Search Range (+/- 20%)
            search_margin = int(avg_zone_h * 0.2)
            scan_start_y = min(h-1, estimated_hairline_y + search_margin)
            scan_end_y = max(0, estimated_hairline_y - search_margin)
            
            if scan_start_y <= scan_end_y:
                 return estimated_hairline_y

            # ROI X coordinates (center strip)
            center_x = int(lm_10.x * w)
            roi_w = 20
            x1 = max(0, center_x - roi_w // 2)
            x2 = min(w, center_x + roi_w // 2)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            strip = gray[scan_end_y:scan_start_y, x1:x2]
            
            if strip.size == 0:
                return estimated_hairline_y
                
            # Calculate vertical gradient
            grad_y = cv2.Sobel(strip, cv2.CV_64F, 0, 1, ksize=3)
            grad_y = np.abs(grad_y)
            
            # Average gradient
            mean_grad = np.mean(grad_y, axis=1)
            mean_grad = cv2.GaussianBlur(mean_grad.reshape(-1, 1), (1, 5), 0)
            
            # Threshold
            threshold = np.max(mean_grad) * 0.25
            
            detected_y = estimated_hairline_y
            found = False
            
            # Scan from bottom up
            for i in range(len(mean_grad) - 1, 0, -1):
                if mean_grad[i] > threshold:
                    detected_y = scan_end_y + i
                    found = True
                    break
            
            return detected_y if found else estimated_hairline_y
            
        except Exception as e:
            print(f"Hairline detection error: {e}")
            # Fallback
            if hasattr(landmarks, 'landmark'):
                landmarks = landmarks.landmark
            lm_brow = landmarks[105]
            lm_nose = landmarks[2]
            zone2_h = abs(lm_nose.y - lm_brow.y) * h
            return int(lm_brow.y * h - zone2_h)

    def calculate_zone_metrics(self, img, landmarks):
        """
        Calculate detailed metrics for face shape analysis.
        
        Metrics based on:
        - FL (Face Length): Top (10) to Chin (152)
        - FW (Forehead Width): 103 <-> 332
        - CW (Cheekbone Width): 234 <-> 454 (widest point)
        - JW (Jaw Width): 132 <-> 361 (Jaw corners/Gonions)
        """
        h, w = img.shape[:2]
        
        # Handle NormalizedLandmarkList
        if hasattr(landmarks, 'landmark'):
            landmarks = landmarks.landmark
            
        if len(landmarks) < 468:
            print(f"Warning: Landmarks list too short ({len(landmarks)}) for zone metrics.")
            return {
                'FL': 0, 'FW': 0, 'CW': 0, 'JW': 0,
                'WLR': 0, 'jaw_angle': 0,
                'z1_h': 0, 'z1_w': 0, 'z2_h': 0, 'z2_w': 0, 'z3_h': 0, 'z3_w': 0, # Keep for compatibility
                'face_height': 0
            }
        
        # Helper for pixel coordinates
        def p(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
            
        def dist(p1, p2):
            return np.linalg.norm(p1 - p2)

        # 1. Face Length (FL)
        # Using landmark 10 (top center) to 152 (chin)
        # Note: 10 is usually top of forehead/hairline approx in mesh
        fl_point_top = p(10)
        fl_point_bot = p(152)
        FL = dist(fl_point_top, fl_point_bot)
        
        # 2. Forehead Width (FW)
        # 103 (left) <-> 332 (right)
        FW = dist(p(103), p(332))
        
        # 3. Cheekbone Width (CW)
        # 234 (left) <-> 454 (right) - typically widest
        CW = dist(p(234), p(454))
        
        # 4. Jaw Width (JW)
        # 132 (left gonion) <-> 361 (right gonion)
        # Alternatively 172 <-> 397 (jawline), but 132/361 are corners
        JW = dist(p(132), p(361))
        
        # 5. Jaw Angle
        # Triangle: Left Jaw (132) -> Chin (152) -> Right Jaw (361)
        # We calculate the angle at the chin? Or the slope?
        # User says: "Jaw angle / slope: çene köşesinin sertliği"
        # "Jaw angle: üçgen oluştur jaw_left, chin, jaw_right — çenenin dar/genişliğini ve açısını analiz etmek için"
        # Usually "Jaw Angle" refers to the angle at the gonion (132), but here user implies angle at chin or general width/height ratio of jaw.
        # Let's calculate the angle at the chin (152) formed by 132-152-361.
        # Wider angle = softer/square/round. Narrow angle = triangle/heart.
        
        p_left = p(132)
        p_chin = p(152)
        p_right = p(361)
        
        v1 = p_left - p_chin
        v2 = p_right - p_chin
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        # 6. Width/Length Ratio (WLR)
        # max(FW, CW, JW) / FL
        max_width = max(FW, CW, JW)
        WLR = max_width / FL if FL > 0 else 0
        
        # Compatibility metrics (for visualization drawing)
        # We map new precise metrics to old keys where appropriate or recalculate
        
        return {
            'FL': FL,
            'FW': FW,
            'CW': CW,
            'JW': JW,
            'WLR': WLR,
            'jaw_angle': angle,
            
            # Legacy/Visualization keys
            'z1_w': FW,
            'z2_w': CW,
            'z3_w': JW,
            'face_height': FL,
            'z1_h': abs(p(10)[1] - p(105)[1]), # Approx forehead height
            'z2_h': abs(p(105)[1] - p(2)[1]),  # Approx mid face
            'z3_h': abs(p(2)[1] - p(152)[1]),  # Approx lower face
        }

    def _detect_face_shape(self, metrics, landmarks, w, h):
        """
        Detect face shape based on user-provided ratios.
        
        Metrics:
        - FL: Face Length
        - FW: Forehead Width
        - CW: Cheekbone Width
        - JW: Jaw Width
        - WLR: Width/Length Ratio
        - Jaw Angle
        """
        FL = metrics.get('FL', 0)
        FW = metrics.get('FW', 0)
        CW = metrics.get('CW', 0)
        JW = metrics.get('JW', 0)
        WLR = metrics.get('WLR', 0)
        angle = metrics.get('jaw_angle', 0)
        
        if FL == 0:
            return {'shape': 'Belirsiz'}
            
        shape = 'Belirsiz'
        
        # Helper for approximate equality (within tolerance)
        def approx(a, b, tol=0.1):
            return abs(a - b) < (b * tol)
            
        # --- CLASSIFICATION LOGIC ---
        
        # 1. DIKDÖRTGEN / UZUN (Rectangular / Oblong)
        # FL >> Widths
        # WLR < 0.55 (User said < 0.4, but that's very extreme. 0.55 is a safer start for "oblong")
        # Let's stick closer to user's "WLR small" logic.
        # Also FW ≈ CW ≈ JW
        if WLR < 0.65 and approx(FW, CW, 0.15) and approx(CW, JW, 0.15):
             shape = 'Dikdörtgen'
             
        # 2. KARE (Square)
        # FL ≈ CW ≈ JW (WLR ≈ 0.8 - 1.0)
        # Jaw corners distinct (Angle usually wider/harder, but here "distinct" means width is maintained)
        elif 0.75 <= WLR <= 1.0 and approx(FW, CW, 0.15) and approx(CW, JW, 0.15):
            shape = 'Kare'
            
        # 3. YUVARLAK (Round)
        # FL ≈ CW (WLR ≈ 0.75 - 0.85)
        # Soft corners -> Wide chin angle? 
        # User: "Jaw angle > 130"
        # Also FL is "small" (relative to what? usually implies WLR is higher, closer to 1)
        elif 0.75 <= WLR <= 1.0 and angle > 125:
             # Round vs Square: Round has softer jaw (wider angle at chin or less defined corners)
             # If widths are similar but angle is wide -> Round
             if approx(CW, JW, 0.2): 
                 shape = 'Yuvarlak'
        
        # 4. KALP / ÜÇGEN (Heart / Triangle)
        # FW > CW > JW
        # Chin narrow
        elif FW > CW and CW > JW:
            shape = 'Ters Üçgen (Kalp)'
            
        # Normal Triangle: JW > CW > FW (Pear shape)
        elif JW > CW and CW > FW:
            shape = 'Üçgen'
            
        # 5. OVAL
        # FL long (WLR ~ 0.6 - 0.7)
        # CW > FW and CW > JW (Cheekbone is widest)
        # Soft chin
        elif CW > FW and CW > JW:
            if 0.55 <= WLR <= 0.75:
                shape = 'Oval'
            else:
                # If WLR is high but cheek is widest -> Round/Oval mix
                shape = 'Oval'
        
        # Fallbacks
        if shape == 'Belirsiz':
            if WLR > 0.8:
                shape = 'Yuvarlak' if angle > 120 else 'Kare'
            else:
                shape = 'Oval' if CW > JW else 'Dikdörtgen'
                
        return {'shape': shape}
    
    def _analyze_forehead(self, metrics, landmarks, w, h):
        """Analyze forehead characteristics"""
        # Height
        z1_h = metrics['z1_h']
        z2_h = metrics['z2_h']
        z3_h = metrics['z3_h']
        
        height_class = 'Normal'
        if z1_h < z2_h and z1_h < z3_h:
            height_class = 'Kısa'
        elif z1_h > z2_h and z1_h > z3_h:
            height_class = 'Yüksek'
            
        # Width
        z1_w = metrics['z1_w']
        z3_w = metrics['z3_w']
        
        width_class = 'Normal'
        if z3_w > 0:
            ratio = z1_w / z3_w
            if ratio > 1.15: width_class = 'Geniş'
            elif ratio < 0.85: width_class = 'Dar'
            
        # Slope (Z-axis)
        lm_top = landmarks[10]
        lm_brow = landmarks[8]
        lm_chin = landmarks[152]
        
        face_h_norm = abs(lm_chin.y - lm_top.y)
        slope_class = 'Düz'
        
        if face_h_norm > 0:
            z_diff = lm_top.z - lm_brow.z
            slope_score = z_diff / face_h_norm
            
            if slope_score > 0.025: slope_class = 'Eğimli'
            elif slope_score < -0.015: slope_class = 'Yuvarlak'
            
        return {
            'width': width_class,
            'height': height_class,
            'slope': slope_class
        }
    
    def _analyze_eyes(self, metrics, landmarks, w, h):
        """Analyze eye characteristics"""
        # Size
        l_top = landmarks[159]; l_bot = landmarks[145]
        r_top = landmarks[386]; r_bot = landmarks[374]
        
        avg_eye_h = (abs(l_bot.y - l_top.y) + abs(r_bot.y - r_top.y)) / 2
        face_h_norm = abs(landmarks[152].y - landmarks[10].y)
        
        size_class = 'Normal'
        if face_h_norm > 0:
            ratio = avg_eye_h / face_h_norm
            if ratio > 0.06: size_class = 'Büyük'
            elif ratio < 0.045: size_class = 'Küçük'
            
        # Spacing
        l_width = abs(landmarks[33].x - landmarks[133].x)
        r_width = abs(landmarks[263].x - landmarks[362].x)
        avg_width = (l_width + r_width) / 2
        icd = abs(landmarks[133].x - landmarks[362].x)
        
        spacing_class = 'Normal'
        if avg_width > 0:
            ratio = icd / avg_width
            if ratio > 1.15: spacing_class = 'Ayrık'
            elif ratio < 0.85: spacing_class = 'Bitişik'
            
        # Slant (Tilt)
        dy_l = landmarks[133].y - landmarks[33].y
        dx_l = abs(landmarks[33].x - landmarks[133].x)
        angle_l = np.degrees(np.arctan(dy_l / dx_l)) if dx_l > 0 else 0
        
        dy_r = landmarks[362].y - landmarks[263].y
        dx_r = abs(landmarks[263].x - landmarks[362].x)
        angle_r = np.degrees(np.arctan(dy_r / dx_r)) if dx_r > 0 else 0
        
        avg_angle = (angle_l + angle_r) / 2
        
        slant_class = 'Düz'
        if avg_angle > 5.0: slant_class = 'Çekik'
        elif avg_angle < -5.0: slant_class = 'Düşük'
        
        # Depth
        z_brow_l = landmarks[107].z; z_eye_l = landmarks[159].z
        z_brow_r = landmarks[336].z; z_eye_r = landmarks[386].z
        diff = ((z_eye_l - z_brow_l) + (z_eye_r - z_brow_r)) / 2
        
        depth_class = 'Normal'
        if diff > 0.05: depth_class = 'Çukur'
        elif diff < -0.04: depth_class = 'Çıkık'
        
        return {
            'size': size_class,
            'slant': slant_class,
            'spacing': spacing_class,
            'depth': depth_class
        }
    
    def _analyze_nose(self, metrics, landmarks, w, h):
        """Analyze nose characteristics"""
        # Length
        y_133 = landmarks[133].y * h
        y_362 = landmarks[362].y * h
        nasion_y = (y_133 + y_362) / 2 - (h * 0.015)
        subnasale_y = landmarks[2].y * h
        nose_len_px = abs(subnasale_y - nasion_y)
        face_height_px = metrics['face_height']
        
        length_class = 'Normal'
        if face_height_px > 0:
            ratio = nose_len_px / face_height_px
            if ratio > 0.35: length_class = 'Uzun'
            elif ratio < 0.29: length_class = 'Kısa'
            
        # Width
        alar_l = landmarks[102]
        alar_r = landmarks[331]
        nose_width = abs(alar_r.x - alar_l.x)
        icd = abs(landmarks[133].x - landmarks[362].x)
        
        width_class = 'Normal'
        if icd > 0:
            ratio = nose_width / icd
            if ratio > 1.15: width_class = 'Geniş'
            elif ratio < 0.85: width_class = 'Dar'
            
        # Tip
        lm_tip = landmarks[1]; lm_base = landmarks[2]
        face_h_norm = abs(landmarks[152].y - landmarks[10].y)
        
        tip_class = 'Normal'
        if face_h_norm > 0:
            diff = (lm_base.y - lm_tip.y) / face_h_norm
            if diff > 0.015: tip_class = 'Kalkık'
            elif diff < -0.015: tip_class = 'Düşük'
            
        return {
            'length': length_class,
            'width': width_class,
            'bridge': 'Düz', # Default
            'tip': tip_class
        }
    
    def _analyze_lips(self, metrics, landmarks, w, h):
        """Analyze lip characteristics"""
        # Width
        lm_mouth_l = landmarks[61]; lm_mouth_r = landmarks[291]
        lm_pupil_l = landmarks[468]; lm_pupil_r = landmarks[473]
        
        width_class = 'Normal'
        if lm_mouth_l.x < lm_pupil_l.x and lm_mouth_r.x > lm_pupil_r.x:
            width_class = 'Geniş'
        elif lm_mouth_l.x > lm_pupil_l.x and lm_mouth_r.x < lm_pupil_r.x:
            width_class = 'Dar'
            
        # Thickness
        lm_ul_top = landmarks[13]; lm_ul_bot = landmarks[14]
        lm_ll_top = landmarks[14]; lm_ll_bot = landmarks[17]
        
        h_upper = abs(lm_ul_bot.y - lm_ul_top.y)
        h_lower = abs(lm_ll_bot.y - lm_ll_top.y)
        face_h_norm = abs(landmarks[152].y - landmarks[10].y)
        
        upper_class = 'Normal'; lower_class = 'Normal'
        if face_h_norm > 0:
            r_u = h_upper / face_h_norm
            r_l = h_lower / face_h_norm
            
            if r_u > 0.035: upper_class = 'Kalın'
            elif r_u < 0.02: upper_class = 'İnce'
            
            if r_l > 0.04: lower_class = 'Kalın'
            elif r_l < 0.025: lower_class = 'İnce'
            
        return {
            'upper_thickness': upper_class,
            'lower_thickness': lower_class,
            'width': width_class
        }
    
    def _analyze_chin(self, metrics, landmarks, w, h):
        """Analyze chin characteristics"""
        # Width
        w_jaw = abs(landmarks[397].x - landmarks[172].x)
        w_cheek = abs(landmarks[454].x - landmarks[234].x)
        
        width_class = 'Normal'
        if w_cheek > 0:
            ratio = w_jaw / w_cheek
            if ratio > 0.9: width_class = 'Geniş'
            elif ratio < 0.7: width_class = 'Dar'
            
        # Prominence
        z1_h = metrics['z1_h']; z2_h = metrics['z2_h']; z3_h = metrics['z3_h']
        face_height = z1_h + z2_h + z3_h
        z3_ratio = z3_h / face_height if face_height > 0 else 0.33
        
        lm_lip_bot = landmarks[17]; lm_chin_tip = landmarks[152]; lm_nose_base = landmarks[2]
        lip_chin_h = abs(lm_chin_tip.y - lm_lip_bot.y)
        z3_h_calc = abs(lm_chin_tip.y - lm_nose_base.y)
        lip_chin_ratio = lip_chin_h / z3_h_calc if z3_h_calc > 0 else 0.5
        
        w_chin_tip = abs(landmarks[377].x - landmarks[148].x)
        w_jaw_base = abs(landmarks[397].x - landmarks[172].x)
        chin_shape_ratio = w_chin_tip / w_jaw_base if w_jaw_base > 0 else 1.0
        
        prom_class = 'Normal'
        if z3_ratio > 0.36 and (lip_chin_ratio < 0.42 or chin_shape_ratio < 0.45):
            prom_class = 'Çıkık'
        elif z3_ratio < 0.30:
            prom_class = 'Geride'
            
        return {
            'width': width_class,
            'prominence': prom_class,
            'dimple': 'Yok'
        }
    
    def _analyze_ears(self, landmarks, w, h):
        """Analyze ears (Limited in 2D)"""
        # Placeholder
        return {
            'prominence': 'Normal',
            'lobe': 'Normal'
        }

    def detect_ear_vertical_bounds(self, img, landmarks, side='right'):
        """
        Detect ear vertical bounds using anatomical positioning.
        MediaPipe's ear landmarks (127, 93, 356, 323) are unreliable.
        Instead, we use the fact that ears typically span from eyebrow to nose base.
        """
        h, w = img.shape[:2]
        
        # Handle NormalizedLandmarkList
        if hasattr(landmarks, 'landmark'):
            landmarks = landmarks.landmark
        
        # Use reliable facial landmarks for ear positioning
        # Ears typically align with eyebrow (top) and nose base (bottom)
        
        try:
            # Get eyebrow and nose positions
            brow_y = int((landmarks[105].y + landmarks[334].y) / 2 * h)  # Average of left and right brow
            nose_base_y = int(landmarks[2].y * h)  # Nose tip
            
            # Adjust for anatomical reality:
            # - Ear top is typically slightly above eyebrow
            # - Ear bottom (lobe) is typically at or slightly below nose base
            ear_top_y = max(0, brow_y - int(h * 0.02))  # 2% above brow
            ear_bot_y = min(h, nose_base_y + int(h * 0.03))  # 3% below nose
            
            # Calculate ear height for validation
            ear_height = ear_bot_y - ear_top_y
            
            # Define landmarks based on side for X position
            if side == 'right':
                # Person's Right Ear (Image Left)
                idx_ref = 234  # Cheek/Ear connection point
            else:
                # Person's Left Ear (Image Right)
                idx_ref = 454
            
            if idx_ref >= len(landmarks):
                return ear_top_y, ear_bot_y, 0.5
            
            # Get reference X position
            ref_x = int(landmarks[idx_ref].x * w)
            
            # Calculate face width for shift amount
            face_width = abs(landmarks[454].x - landmarks[234].x) * w
            shift_amount = int(face_width * 0.18)  # Shift outward to ear location
            
            # Determine scan position
            if side == 'right':
                scan_x = max(0, ref_x - shift_amount)
            else:
                scan_x = min(w - 1, ref_x + shift_amount)
            
            # Now refine using edge detection
            confidence = 0.65  # Base confidence with anatomical positioning
            
            # Define scan range around anatomical estimates
            scan_y_start = max(0, ear_top_y - int(ear_height * 0.3))
            scan_y_end = min(h, ear_bot_y + int(ear_height * 0.3))
            
            if scan_y_end <= scan_y_start:
                return ear_top_y, ear_bot_y, confidence
            
            # Extract vertical scan strip
            strip_width = 8  # Wider strip for better detection
            x_start = max(0, scan_x - strip_width // 2)
            x_end = min(w, scan_x + strip_width // 2)
            
            scan_strip = img[scan_y_start:scan_y_end, x_start:x_end]
            
            if scan_strip.size == 0:
                return ear_top_y, ear_bot_y, confidence
            
            # Convert to grayscale for edge detection
            gray_strip = cv2.cvtColor(scan_strip, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_strip, (5, 5), 1.0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 30, 100)
            
            # Average edge intensity across width
            edge_profile = np.mean(edges, axis=1) / 255.0
            
            # Find transitions
            # Top: First significant edge in upper region
            upper_region = len(edge_profile) // 3
            top_found = False
            for i in range(upper_region):
                if edge_profile[i] > 0.15:  # Edge threshold
                    refined_top = scan_y_start + i
                    # Only update if within reasonable range
                    if abs(refined_top - ear_top_y) < ear_height * 0.4:
                        ear_top_y = refined_top
                        confidence = 0.85
                        top_found = True
                    break
            
            # Bottom: Last significant edge in lower region
            lower_region_start = len(edge_profile) * 2 // 3
            bot_found = False
            for i in range(len(edge_profile) - 1, lower_region_start, -1):
                if edge_profile[i] > 0.15:
                    refined_bot = scan_y_start + i
                    # Only update if within reasonable range
                    if abs(refined_bot - ear_bot_y) < ear_height * 0.4:
                        ear_bot_y = refined_bot
                        if top_found:
                            confidence = 0.95
                        else:
                            confidence = 0.80
                        bot_found = True
                    break
            
            # Final validation: ear height should be reasonable
            final_height = ear_bot_y - ear_top_y
            if final_height < h * 0.05 or final_height > h * 0.25:
                # Revert to anatomical estimates
                # Recalculate
                brow_y = int((landmarks[105].y + landmarks[334].y) / 2 * h)
                nose_base_y = int(landmarks[2].y * h)
                ear_top_y = max(0, brow_y - int(h * 0.02))
                ear_bot_y = min(h, nose_base_y + int(h * 0.03))
                confidence = 0.60
            
            return ear_top_y, ear_bot_y, confidence
            
        except Exception as e:
            print(f"Ear detection error ({side}): {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: Use anatomical estimates
            try:
                brow_y = int((landmarks[105].y + landmarks[334].y) / 2 * h)
                nose_base_y = int(landmarks[2].y * h)
                ear_top_y = max(0, brow_y - int(h * 0.02))
                ear_bot_y = min(h, nose_base_y + int(h * 0.03))
                return ear_top_y, ear_bot_y, 0.5
            except:
                return int(h * 0.3), int(h * 0.6), 0.0

    def _default_annotations(self):
        """Return default annotations structure"""
        return {
            'face_shape': {'shape': 'Belirsiz'},
            'forehead': {'width': 'Normal', 'height': 'Normal', 'slope': 'Düz'},
            'eyes': {'size': 'Normal', 'spacing': 'Normal', 'slant': 'Düz', 'depth': 'Normal'},
            'nose': {'length': 'Normal', 'width': 'Normal', 'tip': 'Normal', 'hump': 'Düz'},
            'lips': {'width': 'Normal', 'upper_thickness': 'Normal', 'lower_thickness': 'Normal'},
            'chin': {'width': 'Normal', 'prominence': 'Normal', 'dimple': 'Yok'},
            'ears': {'prominence': 'Normal', 'lobe': 'Normal'}
        }


class AnnotationManager:
    """Manage annotation storage and retrieval using Database"""
    
    def __init__(self):
        # Import here to avoid circular imports if any
        try:
            from desktop_app.database import Database
            self.db = Database()
        except ImportError:
            # Fallback for scripts running from root
            import sys
            sys.path.append(os.getcwd())
            from desktop_app.database import Database
            self.db = Database()
            
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.processed_dir = os.path.join(self.base_dir, "dataset", "processed")
        self.annotations_dir = os.path.join(self.base_dir, "dataset", "annotations")
    
    def migrate_from_files(self):
        """Migrate existing files to database"""
        print("Starting migration to database...")
        count = 0
        
        # 1. Load all images from processed directory
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.processed_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for filename in os.listdir(split_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(split_dir, filename)
                    
                    # Check if already annotated
                    annotation_path = os.path.join(self.annotations_dir, os.path.splitext(filename)[0] + '_annotation.json')
                    
                    # Read image
                    img = cv2.imread(image_path)
                    if img is None:
                        continue
                        
                    # Save to DB (pending) - save_training_data handles INSERT OR IGNORE
                    self.db.save_training_data(filename, img, split)
                    
                    # If annotation exists, update it
                    if os.path.exists(annotation_path):
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            self.db.update_annotation(filename, data)
                    
                    count += 1
                    
        print(f"Migration completed. Processed {count} images.")
        # Refresh ID list after migration
        self.refresh_ids()
        return count

    def refresh_ids(self):
        """Refresh the list of available IDs"""
        self.all_ids = self.db.get_all_annotation_ids()

    def get_total_count(self):
        """Get total number of images"""
        if not hasattr(self, 'all_ids'):
            self.refresh_ids()
        return len(self.all_ids)

    def get_image_at_index(self, index):
        """Get image data at specific index"""
        if not hasattr(self, 'all_ids'):
            self.refresh_ids()
            
        if 0 <= index < len(self.all_ids):
            annot_id = self.all_ids[index]
            return self.db.get_annotation_by_id(annot_id)
        return None
    
    def save_annotation(self, image_name, annotations):
        """Save annotations to DB"""
        self.db.update_annotation(image_name, annotations)
        
    def get_status(self):
        """Get annotation progress"""
        return self.db.get_annotation_status()
    
    def cleanup_files(self):
        """Delete files after successful migration (Optional/Manual trigger)"""
        # This is destructive, so we'll leave it as a manual method for now
        pass
