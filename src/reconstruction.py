import mediapipe as mp
import cv2
import numpy as np

class FaceReconstructor:
    def __init__(self, static_image_mode=False, max_num_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Processes a single frame and returns 3D landmarks.
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0] # Return first face found
        return None

    def get_3d_points(self, landmarks, image_shape):
        """
        Converts normalized landmarks to 3D coordinates (x, y, z).
        """
        h, w, c = image_shape
        points = []
        
        # Handle different input types
        # If landmarks has 'landmark' attribute (e.g. NormalizedLandmarkList), use it
        if hasattr(landmarks, 'landmark'):
            lms = landmarks.landmark
        else:
            # Otherwise assume it's already iterable (list or RepeatedCompositeFieldContainer)
            lms = landmarks
            
        for landmark in lms:
            # x and y are normalized to [0.0, 1.0] by the image width and height.
            # z is represented as the landmark depth with the depth at the center of the head being the origin,
            # and the smaller the value the closer the landmark is to the camera.
            # z uses roughly the same scale as x.
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cz = landmark.z * w # Scale z by width to keep aspect ratio roughly correct
            points.append([cx, cy, cz])
        return np.array(points)
