import numpy as np
import cv2
from src.geometry import euclidean_distance, midpoint, calculate_angle

class FaceFeatures:
    def __init__(self, landmarks_3d, frame=None, side_landmarks=None, side_frame=None, annotations=None):
        """
        :param landmarks_3d: Numpy array of shape (468, 3) containing 3D coordinates.
        :param frame: Optional BGR image for texture analysis.
        :param side_landmarks: Optional 3D landmarks for side profile.
        :param side_frame: Optional BGR image for side profile.
        :param annotations: Optional dict from AutoAnnotator (forehead, eyes, nose, lips, chin etc.)
        """
        self.landmarks = landmarks_3d
        self.frame = frame
        self.side_landmarks = side_landmarks
        self.side_frame = side_frame
        self.annotations = annotations or {}
        self.metrics = {}
        self._calculate_metrics()
        self._determine_face_shape()
        
        if self.side_landmarks is not None:
            self._analyze_side_profile()

    def get_metric(self, name):
        """Returns a specific metric value."""
        return self.metrics.get(name, 0.0)

    def _calculate_metrics(self):
        """
        Calculates raw geometric metrics from landmarks.
        MediaPipe Landmark Indices (approximate):
        - Chin: 152
        - Forehead Top: 10
        - Left Cheek: 234
        - Right Cheek: 454
        - Left Eye Outer: 33
        - Right Eye Outer: 263
        - Left Eye Inner: 133
        - Right Eye Inner: 362
        - Nose Tip: 1
        """
        
        # Points of interest
        chin = self.landmarks[152]
        forehead_top = self.landmarks[10]
        left_cheek = self.landmarks[234]
        right_cheek = self.landmarks[454]
        
        left_eye_outer = self.landmarks[33]
        right_eye_outer = self.landmarks[263]
        left_eye_inner = self.landmarks[133]
        right_eye_inner = self.landmarks[362]
        
        # 1. Face Dimensions
        face_height = euclidean_distance(forehead_top, chin)
        face_width = euclidean_distance(left_cheek, right_cheek)
        self.metrics['face_width'] = face_width
        self.metrics['face_height'] = face_height
        self.metrics['face_wh_ratio'] = face_width / face_height if face_height > 0 else 0

        # 2. Jawline (Approximate using cheek to chin path)
        # Jaw width is often measured at the gonion (angle of jaw), 
        # but MediaPipe doesn't have a perfect bone landmark. 
        # We use a lower cheek point as proxy.
        left_jaw = self.landmarks[58]  # Approx
        right_jaw = self.landmarks[288] # Approx
        jaw_width = euclidean_distance(left_jaw, right_jaw)
        self.metrics['jaw_width'] = jaw_width
        self.metrics['jaw_face_width_ratio'] = jaw_width / face_width if face_width > 0 else 0

        # 3. Eyes
        inter_ocular_distance = euclidean_distance(left_eye_inner, right_eye_inner)
        left_eye_width = euclidean_distance(left_eye_outer, left_eye_inner)
        right_eye_width = euclidean_distance(right_eye_outer, right_eye_inner)
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        
        self.metrics['inter_ocular_distance'] = inter_ocular_distance
        self.metrics['avg_eye_width'] = avg_eye_width
        self.metrics['eye_spacing_ratio'] = inter_ocular_distance / avg_eye_width if avg_eye_width > 0 else 0
        
        # Eye Size (Relative to Face Width)
        self.metrics['eye_size_ratio'] = avg_eye_width / face_width if face_width > 0 else 0

        # 4. Forehead (Approximate)
        # Forehead width can be estimated between temples
        left_temple = self.landmarks[21] 
        right_temple = self.landmarks[251] 
        forehead_width = euclidean_distance(left_temple, right_temple)
        
        # Forehead height (approximate: top of head to eyebrows)
        # Note: Hairline detection is hard without segmentation. Using mesh top (10) to brow center (168)
        brow_center = self.landmarks[168]
        forehead_height = euclidean_distance(forehead_top, brow_center)
        
        self.metrics['forehead_width'] = forehead_width
        self.metrics['forehead_height'] = forehead_height
        self.metrics['forehead_jaw_ratio'] = forehead_width / jaw_width if jaw_width > 0 else 0
        self.metrics['forehead_h_w_ratio'] = forehead_height / forehead_width if forehead_width > 0 else 0

        # 5. Nose
        nose_tip = self.landmarks[1]
        nose_top = self.landmarks[168] # Glabella
        nose_bottom = self.landmarks[2]
        nose_left = self.landmarks[279] # Alar base
        nose_right = self.landmarks[49] # Alar base
        
        nose_length = euclidean_distance(nose_top, nose_bottom)
        nose_width = euclidean_distance(nose_left, nose_right)
        
        # Nose Bridge Width (Approx between 197 and 236 - near mid bridge)
        # Or just use distance between inner eye corners as a reference for root width?
        # Let's use landmarks on the nose ridge sides: 196 (left), 419 (right) approx?
        # MediaPipe nose landmarks are dense. 
        # 168 is top. 6 is mid. 197/236 are nearby.
        # Let's use 193 (left) and 417 (right) for bridge width.
        nose_bridge_left = self.landmarks[193]
        nose_bridge_right = self.landmarks[417]
        nose_bridge_width = euclidean_distance(nose_bridge_left, nose_bridge_right)
        
        self.metrics['nose_length'] = nose_length
        self.metrics['nose_width'] = nose_width
        self.metrics['nose_bridge_width'] = nose_bridge_width
        self.metrics['nose_ratio'] = nose_width / nose_length if nose_length > 0 else 0
        self.metrics['nose_bridge_ratio'] = nose_bridge_width / nose_width if nose_width > 0 else 0
        
        # 6. Lips
        upper_lip_top = self.landmarks[13]
        upper_lip_bottom = self.landmarks[14]
        lower_lip_top = self.landmarks[14] # Same point, inner
        lower_lip_bottom = self.landmarks[17]
        mouth_left = self.landmarks[61]
        mouth_right = self.landmarks[291]
        
        mouth_width = euclidean_distance(mouth_left, mouth_right)
        upper_lip_thickness = euclidean_distance(upper_lip_top, upper_lip_bottom)
        lower_lip_thickness = euclidean_distance(lower_lip_top, lower_lip_bottom)
        
        self.metrics['mouth_width'] = mouth_width
        self.metrics['upper_lip_thickness'] = upper_lip_thickness
        self.metrics['lower_lip_thickness'] = lower_lip_thickness
        self.metrics['lip_ratio'] = upper_lip_thickness / lower_lip_thickness if lower_lip_thickness > 0 else 0

        # 7. Eyebrows (Shape)
        # Check if arched or straight by comparing midpoint height to endpoints
        left_brow_left = self.landmarks[46]
        left_brow_right = self.landmarks[55]
        left_brow_mid = self.landmarks[52] # Arch point
        
        # Simple curvature check: distance of mid point to the line connecting left and right
        # For simplicity, just check relative Y positions (lower Y is higher on face in image coords usually, 
        # but in 3D world coords, Y might be up or down depending on convention. 
        # MediaPipe Z is depth. Y is down in 2D, but let's use the 3D points directly.)
        # We'll use the angle or simple height difference.
        
        # Let's use a simple metric: Arch Height
        brow_base_y = (left_brow_left[1] + left_brow_right[1]) / 2
        brow_arch_y = left_brow_mid[1]
        # In image coords, smaller Y is higher. So if arch < base, it's arched.
        self.metrics['eyebrow_arch'] = brow_base_y - brow_arch_y # Positive means arched up

        # 8. Facial Zones (The 3 Regions)
        # Region 1: Hairline (10) to Eyebrow Top (approx 105/334 or mid brow)
        # Region 2: Eyebrow Top to Nose Bottom (2)
        # Region 3: Nose Bottom (2) to Chin (152)
        
        # Note: Point 10 is top of mesh, not necessarily hairline, but best proxy.
        # Using average brow height for boundary 1-2
        brow_avg_y = (self.landmarks[105][1] + self.landmarks[334][1]) / 2 # Approx brow top
        nose_bottom_y = nose_bottom[1]
        chin_y = chin[1]
        top_y = self.landmarks[10][1]
        
        # Calculate vertical distances (y differences)
        # Note: In 3D space, we should use Euclidean distance, but for vertical zones, Y-diff is often used in 2D physiognomy.
        # Let's use Euclidean distance between projected points on the vertical axis to be safe, or just direct distance if face is frontal.
        # Assuming frontal face for now, Y diff is okay.
        
        r1 = abs(brow_avg_y - top_y)
        r2 = abs(nose_bottom_y - brow_avg_y)
        r3 = abs(chin_y - nose_bottom_y)
        
        total_h = r1 + r2 + r3
        self.metrics['zone_1_ratio'] = r1 / total_h if total_h > 0 else 0
        self.metrics['zone_2_ratio'] = r2 / total_h if total_h > 0 else 0
        self.metrics['zone_3_ratio'] = r3 / total_h if total_h > 0 else 0
        
        # 9. Cheekbones
        # Prominence: Ratio of cheek width to jaw width and temple width
        # If Cheek > Jaw and Cheek > Temple -> Prominent
        self.metrics['cheek_jaw_ratio'] = face_width / jaw_width if jaw_width > 0 else 0
        self.metrics['cheek_temple_ratio'] = face_width / forehead_width if forehead_width > 0 else 0

        # 10. Chin Shape
        # Calculate angle at chin (152) using jaw points (58, 288)
        # Sharp angle (< 110?) -> Pointy. Wide angle (> 130?) -> Square/Broad.
        # Using geometry.calculate_angle(p1, p2, p3) -> angle at p2
        chin_angle = calculate_angle(left_jaw, chin, right_jaw)
        self.metrics['chin_angle'] = chin_angle

        # 11. Asymmetry (Forehead)
        # Image Left (21) is Person's Right (Material)
        # Image Right (251) is Person's Left (Spiritual)
        # We need distance from midline. Midline defined by nose top (168) and chin (152).
        # Simple approximation: |x_mid - x_point| if face is upright.
        # Better: Distance from point to line.
        
        # Vector of midline
        mid_vector = chin - nose_top
        # Normalize
        mid_len = np.linalg.norm(mid_vector)
        if mid_len > 0:
            mid_unit = mid_vector / mid_len
            
            # Vector from nose_top to temples
            v_right_temple = left_temple - nose_top # Image Left = Person Right
            v_left_temple = right_temple - nose_top # Image Right = Person Left
            
            # Project onto midline to find perpendicular distance (rejection)
            # dist = ||v - (v . u) * u||
            proj_right = np.dot(v_right_temple, mid_unit) * mid_unit
            dist_right = np.linalg.norm(v_right_temple - proj_right) # Person Right (Maddi)
            
            proj_left = np.dot(v_left_temple, mid_unit) * mid_unit
            dist_left = np.linalg.norm(v_left_temple - proj_left) # Person Left (Manevi)
            
            self.metrics['forehead_right_width'] = dist_right
            self.metrics['forehead_left_width'] = dist_left
            self.metrics['forehead_asymmetry'] = dist_right - dist_left # Positive = Right (Maddi) dominant
        else:
            self.metrics['forehead_asymmetry'] = 0

        # 12. Eye Depth (Protruding vs Deep Set)
        # Compare Z of eye pupil/center to Z of eyebrow/cheek.
        # Using simple Z difference between Eye Outer (33/263) and Eyebrow Center (52/282)
        # Note: MediaPipe Z: smaller value = closer to camera.
        # Deep set: Eye Z > Brow Z (Eye is further away)
        # Protruding: Eye Z ~ Brow Z
        
        # Left Eye
        l_eye_z = left_eye_outer[2]
        l_brow_z = self.landmarks[52][2]
        eye_depth = l_eye_z - l_brow_z # Positive = Eye is behind brow (Deep)
        self.metrics['eye_depth'] = eye_depth

        # 13. Lip Corners (Happy vs Sad)
        # Compare Y of corners (61, 291) to Y of mouth center (0 - upper lip middle approx or 13)
        mouth_center_y = (upper_lip_top[1] + lower_lip_bottom[1]) / 2
        mouth_corners_avg_y = (mouth_left[1] + mouth_right[1]) / 2
        
        # Y increases downwards. 
        # If Corners Y > Center Y -> Corners are lower -> Sad
        self.metrics['mouth_corner_drop'] = mouth_corners_avg_y - mouth_center_y

        # 14. Nose Tip Shape (Sharpness)
        nose_tip_angle = calculate_angle(nose_left, nose_tip, nose_right)
        self.metrics['nose_tip_angle'] = nose_tip_angle

        # 15. Head Pose (Yaw, Pitch)
        # Estimate using Nose Tip(1) vs Face Center/Ears
        # Simple Yaw: Relative X position of nose tip between cheekbones (234, 454)
        # Range [-1, 1]. 0 = Center.
        left_cheek_x = left_cheek[0]
        right_cheek_x = right_cheek[0]
        nose_x = nose_tip[0]
        
        # Normalize nose x between cheeks
        if right_cheek_x != left_cheek_x:
            # (val - min) / (max - min) -> 0..1
            # Remap to -1..1
            face_width_x = right_cheek_x - left_cheek_x
            relative_nose_x = (nose_x - left_cheek_x) / face_width_x
            yaw_score = (relative_nose_x - 0.5) * 2 # -1 (Left) to 1 (Right)
        else:
            yaw_score = 0
            
        self.metrics['pose_yaw'] = yaw_score
        
        # 16. Profile Features (Only valid if Yaw is significant, e.g. > 0.5)
        # Forehead Slope: Angle between Brow(105) - Forehead Top(10) - Vertical?
        # Or simply Z difference between Top and Brow relative to Y difference.
        # In profile, we look at X-Y plane mostly? No, 3D points are better.
        # Slope: (Z_top - Z_brow) / (Y_top - Y_brow)
        # If Top is further back (Z higher) than Brow -> Sloped.
        # If Top is same Z as Brow -> Vertical/Straight.
        
        # Note: Z increases away from camera? 
        # MediaPipe: Z is depth. Negative is closer? 
        # Actually, usually Z=0 at center, negative is closer to camera.
        # Let's check relative Z.
        # If Forehead Top Z > Brow Z (Top is further away) -> Sloped.
        
        z_diff_forehead = forehead_top[2] - self.landmarks[105][2]
        self.metrics['forehead_slope'] = z_diff_forehead # Positive = Sloped Back

        # 18. Forehead Shape (Round vs Flat/Square)
        # Compare Top Center (10) height to Top Sides (67, 297 - High forehead sides)
        # If Center is significantly higher than sides -> Rounded/Tapered (Oval/Lucky?)
        # If Center is close to sides -> Flat/Square (Broad)
        
        # Y coordinates (smaller is higher)
        top_center_y = self.landmarks[10][1]
        left_top_y = self.landmarks[67][1]
        right_top_y = self.landmarks[297][1]
        avg_side_y = (left_top_y + right_top_y) / 2
        
        # Difference. Positive = Center is higher (smaller Y) than sides
        # We need to normalize by face height or width.
        diff = avg_side_y - top_center_y
        self.metrics['forehead_roundness'] = diff / self.metrics['face_height'] if self.metrics['face_height'] > 0 else 0
        
        # 19. Ear Position (Height)
        # Using landmarks 234 (Left Ear/Cheek edge) and 454 (Right Ear/Cheek edge) as proxies for Ear Tragus/Mid.
        # Compare Y of Ear(234) to Y of Eye Outer(33) and Nose Tip(1).
        # Note: Y increases downwards.
        # High Ear: Ear Y < Eye Y (Higher on face)
        # Low Ear: Ear Y > Nose Y (Lower on face)
        
        # Let's measure relative position of Ear(234) between Eye(33) and Mouth(61).
        # Normalized position: (EarY - EyeY) / (MouthY - EyeY)
        # 0 = At Eye Level. 1 = At Mouth Level.
        
        left_ear_y = self.landmarks[234][1]
        left_eye_y = left_eye_outer[1]
        left_mouth_y = mouth_left[1]
        
        denom = left_mouth_y - left_eye_y
        if denom != 0:
            ear_pos_score = (left_ear_y - left_eye_y) / denom
        else:
            ear_pos_score = 0.5 # Default middle
            
        self.metrics['ear_position'] = ear_pos_score

        # 17. Texture Analysis (Forehead Wrinkles)
        # Requires self.frame
        if self.frame is not None:
            self._analyze_texture()
        else:
            self.metrics['forehead_lines'] = 0

    def _determine_face_shape(self):
        """Yüz şeklini wh_ratio, jaw/forehead oranı ve çene açısına göre belirler."""
        wh = self.metrics.get('face_wh_ratio', 1.0)
        jaw_face = self.metrics.get('jaw_face_width_ratio', 0.8)
        forehead_jaw = self.metrics.get('forehead_jaw_ratio', 1.0)
        chin_angle = self.metrics.get('chin_angle', 120)

        if wh < 0.75:
            self.face_shape = 'Uzun'
        elif wh > 1.05:
            self.face_shape = 'Geniş'
        elif chin_angle < 100:
            self.face_shape = 'Kalp'
        elif jaw_face > 0.9 and forehead_jaw < 1.05:
            self.face_shape = 'Kare'
        elif forehead_jaw > 1.15:
            self.face_shape = 'Üçgen (Yukarı)'
        elif 0.85 <= wh <= 1.05:
            self.face_shape = 'Oval'
        else:
            self.face_shape = 'Yuvarlak'

    def _analyze_texture(self):
        # Define Forehead ROI
        # Top: 10, Bottom: 105 (Brow center), Left: 21, Right: 251
        # We need 2D pixel coordinates for ROI extraction.
        # The landmarks passed here are 3D (x,y,z) where x,y are in pixels (from reconstruction.py).
        
        try:
            x1 = int(self.landmarks[21][0])
            x2 = int(self.landmarks[251][0])
            y1 = int(self.landmarks[10][1])
            y2 = int(self.landmarks[105][1])
            
            # Margin
            h, w, _ = self.frame.shape
            x1 = max(0, x1); x2 = min(w, x2)
            y1 = max(0, y1); y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:
                roi = self.frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Edge detection for lines
                # Canny might be too noisy. Let's use Sobel Y for horizontal lines.
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                abs_sobel_y = np.absolute(sobel_y)
                scaled_sobel = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
                
                # Threshold to find strong lines
                _, thresh = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)
                
                # Calculate density of lines
                line_density = np.sum(thresh) / 255 / (thresh.size)
                self.metrics['forehead_lines'] = line_density
            else:
                self.metrics['forehead_lines'] = 0
        except Exception as e:
            print(f"Texture analysis failed: {e}")
            self.metrics['forehead_lines'] = 0


    def _analyze_side_profile(self):
        """
        Analyzes side profile for head shape (Cranium vs Face).
        """
        # We need to find the Ear, Nose Tip, and Back of Head.
        # In side profile, landmarks might be rotated.
        # Assuming side_landmarks are from a separate detection on the side image.
        
        # Key Landmarks (Side View - Approximate):
        # Nose Tip: 1
        # Ear (Tragus): 234 (Left) or 454 (Right). We'll check which one is visible/detected.
        # Actually, MediaPipe always returns 468 points. We need to know which side is facing camera.
        # We can check z-depth or just assume the user followed instructions.
        # Let's use the wider side.
        
        # Convert landmarks if they are MediaPipe objects
        if hasattr(self.side_landmarks, 'landmark'):
            lm = [[l.x, l.y, l.z] for l in self.side_landmarks.landmark]
        else:
            lm = self.side_landmarks
        
        # 1. Nose Tip (1)
        nose_tip = lm[1]
        
        # 2. Ear
        # Check visibility or Z. Let's assume Left Side Profile (User looking Right) -> Left Ear (234) visible.
        # Or Right Side Profile (User looking Left) -> Right Ear (454) visible.
        # We can check x-coords. If Nose(1) X > Ear(234) X -> Looking Right?
        # Let's calculate both and take the one that makes sense (Nose should be far from Ear).
        ear_left = lm[234]
        ear_right = lm[454]
        
        dist_left = euclidean_distance(nose_tip, ear_left)
        dist_right = euclidean_distance(nose_tip, ear_right)
        
        if dist_left < dist_right: # Right ear is further? Maybe looking left.
            ear = ear_right
            # Looking Left: Back is to the Right.
            direction = "left"
        else:
            ear = ear_left
            # Looking Right: Back is to the Left.
            direction = "right"
            
        # 3. Back of Head
        # MediaPipe Face Mesh DOES NOT cover the back of the head.
        # We MUST use the side_frame and image processing to find the skull contour.
        
        if self.side_frame is None:
            return
            
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.side_frame, cv2.COLOR_BGR2GRAY)
            # Threshold to get silhouette (assuming simple background)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return
                
            # Get largest contour (the head)
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            
            # Map landmarks to pixel coords (normalized 0-1 -> pixel)
            sh, sw = self.side_frame.shape[:2]
            ear_px = (int(ear[0] * sw), int(ear[1] * sh))
            nose_px = (int(nose_tip[0] * sw), int(nose_tip[1] * sh))
            
            # Calculate Front Face Depth (Nose to Ear X distance)
            front_depth = abs(nose_px[0] - ear_px[0])
            
            # Calculate Back Head Depth (Ear to Back of Skull X distance)
            # If looking Right (Nose > Ear), Back is Min X of contour.
            # If looking Left (Nose < Ear), Back is Max X of contour.
            
            if nose_px[0] > ear_px[0]: # Looking Right
                back_x = x # Min X of bounding box
                back_depth = abs(ear_px[0] - back_x)
            else: # Looking Left
                back_x = x + w # Max X of bounding box
                back_depth = abs(back_x - ear_px[0])
                
            # Ratio: Back / Front
            # Normal is maybe 1.0?
            # Big Head / Intellectual: Back > Front significantly?
            cranium_ratio = back_depth / front_depth if front_depth > 0 else 0
            self.metrics['cranium_ratio'] = cranium_ratio
            
            # Head Height / Width
            # Width = front_depth + back_depth
            # Height = h (bounding box height)
            head_width = front_depth + back_depth
            head_height = h
            self.metrics['head_hw_ratio'] = head_height / head_width if head_width > 0 else 0
            
            # Top Head Shape (Vertex)
            # Check if top is flat or pointy.
            # Analyze contour top points.
            # ROI: Top 20% of bounding box.
            
        except Exception as e:
            print(f"Side profile analysis failed: {e}")
