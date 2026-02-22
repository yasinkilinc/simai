import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_color = (0, 255, 0) # Green
        self.landmark_color = (0, 0, 255) # Red

    def draw_landmarks(self, image, landmarks):
        """
        Draws 3D landmarks on the image (projected to 2D).
        """
        h, w, _ = image.shape
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(image, (x, y), 1, self.landmark_color, -1)
        return image

    def draw_analysis(self, image, report):
        """
        Draws the face shape and top traits on the image.
        """
        h, w, _ = image.shape
        
        # Draw Face Shape
        shape_text = f"Sekil: {report['face_shape']}"
        cv2.putText(image, shape_text, (20, 40), self.font, 1.0, (255, 255, 0), 2)
        
        # Draw Positive Traits (Top 3)
        y_offset = 80
        cv2.putText(image, "[+] Olumlu:", (20, y_offset), self.font, self.font_scale, (0, 255, 0), 2)
        y_offset += 25
        for trait in report['analysis']['positive'][:3]:
            # Extract just the trait name (before :)
            short_trait = trait.split(":")[0]
            cv2.putText(image, f"- {short_trait}", (30, y_offset), self.font, self.font_scale, (0, 255, 0), 1)
            y_offset += 25
            
        # Draw Negative Traits (Top 3)
        y_offset += 10
        cv2.putText(image, "[-] Dikkat:", (20, y_offset), self.font, self.font_scale, (0, 0, 255), 2)
        y_offset += 25
        for trait in report['analysis']['negative'][:3]:
            short_trait = trait.split(":")[0]
            cv2.putText(image, f"- {short_trait}", (30, y_offset), self.font, self.font_scale, (0, 0, 255), 1)
            y_offset += 25
            
        return image

    def save_image(self, image, path="analysis_result.jpg"):
        cv2.imwrite(path, image)
