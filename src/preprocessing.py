import cv2
import numpy as np

class FacePreprocessor:
    """Helper class for face preprocessing and alignment"""
    
    def __init__(self):
        self.face_detection = None
        self.face_mesh = None
        self.mp_face_detection = None
        self.mp_face_mesh = None
        self._init_mediapipe()
        
    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            # Face detection for bounding box
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range
                min_detection_confidence=0.5
            )
            # Face mesh for eye alignment  
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except ImportError:
            print("⚠️ Mediapipe kurulu değil! Yüz hizalama devre dışı.")
        except Exception as e:
            print(f"⚠️ Mediapipe başlatılamadı: {e}")

    def align_face(self, img, target_size=None):
        """
        Align face so eyes are horizontal.
        Matches logic from data_preprocessing_view.py
        """
        if self.face_detection is None or self.face_mesh is None:
            return img
        
        h, w = img.shape[:2]
        
        # Convert to RGB for Mediapipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Detect face bounding box
        detection_results = self.face_detection.process(rgb_img)
        if not detection_results.detections:
            return img  # No face detected
        
        detection = detection_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert to pixel coordinates
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bbox_w = int(bbox.width * w)
        bbox_h = int(bbox.height * h)
        
        # 2. Add padding (Matches data_preprocessing_view.py: 1.0 top, 0.5 bottom, 0.4 side)
        padding_top = int(bbox_h * 1.00)
        padding_bottom = int(bbox_h * 0.50)
        padding_side = int(bbox_w * 0.40)
        
        # Calculate expanded bounding box
        x1 = max(0, x - padding_side)
        y1 = max(0, y - padding_top)
        x2 = min(w, x + bbox_w + padding_side)
        y2 = min(h, y + bbox_h + padding_bottom)
        
        # 3. Get eye positions for rotation
        mesh_results = self.face_mesh.process(rgb_img)
        if not mesh_results.multi_face_landmarks:
            # No landmarks, just crop
            cropped = img[y1:y2, x1:x2]
            if target_size:
                return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            return cropped
        
        landmarks = mesh_results.multi_face_landmarks[0]
        
        # Get eye centers
        left_eye_outer = landmarks.landmark[33]
        left_eye_inner = landmarks.landmark[133]
        right_eye_inner = landmarks.landmark[362]
        right_eye_outer = landmarks.landmark[263]
        
        left_eye = np.array([
            int((left_eye_outer.x + left_eye_inner.x) / 2 * w),
            int((left_eye_outer.y + left_eye_inner.y) / 2 * h)
        ])
        right_eye = np.array([
            int((right_eye_inner.x + right_eye_outer.x) / 2 * w),
            int((right_eye_inner.y + right_eye_outer.y) / 2 * h)
        ])
        
        # 4. Calculate rotation angle
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Get center for rotation
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # 5. Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
        
        # 6. Crop the padded region
        cropped = rotated[y1:y2, x1:x2]
        
        # 7. Resize if target size is given
        if target_size:
            aligned = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
            return np.ascontiguousarray(aligned)
            
        return np.ascontiguousarray(cropped)
