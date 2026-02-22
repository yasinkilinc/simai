import cv2
import numpy as np
import logging
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QPixmap, QImage

from PySide6.QtCore import Qt, QPoint, QBuffer

class Visualizer:
    """Helper class for visualizing facial annotations and measurements"""
    
    @staticmethod
    def draw_measurements(pixmap, landmarks, annotator, original_image=None):
        """
        Draw facial zones and measurements on the pixmap.
        
        Args:
            pixmap (QPixmap): The image to draw on.
            landmarks (list): List of landmarks (MediaPipe format).
            annotator (AutoAnnotator): Instance of AutoAnnotator for calculations.
            original_image (np.array, optional): Original CV2 image (BGR). If provided, avoids expensive conversion.
            
        Returns:
            QPixmap: The modified pixmap (drawing is done in-place on the painter, but we return for convenience).
        """
        logging.info(f"DEBUG: Visualizer.draw_measurements called. Pixmap: {pixmap.width()}x{pixmap.height()}")
        
        # Handle NormalizedLandmarkList or list
        lms = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        if len(lms) > 0:
            lm0 = lms[0]
            # Check if lm0 is object or dict
            if hasattr(lm0, 'x'):
                logging.info(f"DEBUG: First landmark (obj): x={lm0.x:.4f}, y={lm0.y:.4f}")
            elif isinstance(lm0, dict):
                logging.info(f"DEBUG: First landmark (dict): x={lm0.get('x'):.4f}, y={lm0.get('y'):.4f}")
        
        print(f"DEBUG: Visualizer.draw_measurements called. Landmarks len: {len(lms)}")
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = pixmap.width()
        h = pixmap.height()
        
        # Helper to get point
        def p(idx):
            # Handle NormalizedLandmarkList
            lms = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
            if idx >= len(lms):
                return 0, 0
            lm = lms[idx]
            return int(lm.x * w), int(lm.y * h)
            
        # Dynamic line width and font size
        line_width = max(2, int(w / 400))
        font_size = max(12, int(w / 60))
        
        # Fonts and Pens
        pen_zone = QPen(QColor(255, 0, 0, 255), line_width, Qt.DashLine) # Red dashed for zones
        pen_measure = QPen(QColor(0, 255, 0, 255), line_width) # Green for features
        
        # Larger font for readability
        font = QFont("Arial", font_size, QFont.Bold)
        painter.setFont(font)
        
        # Helper to draw text with background
        def draw_label(x, y, text, align=Qt.AlignLeft):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            # Add padding
            rect.adjust(-5, -2, 5, 2)
            rect.moveCenter(QPoint(x, y) if align == Qt.AlignCenter else QPoint(x + rect.width()//2, y))
            
            # Draw bg
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(rect, 4, 4)
            
            # Draw text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignCenter, text)

        # --- SMART LABEL SYSTEM ---
        # Color palette for different measurement categories
        COLOR_PALETTE = {
            'zone': QColor(255, 0, 0, 255),         # Red - Zones
            'nose': QColor(255, 215, 0, 255),       # Gold - Nose
            'eye': QColor(0, 255, 255, 255),        # Cyan - Eyes
            'lip': QColor(255, 0, 255, 255),        # Magenta - All Lip measurements
            'ear': QColor(147, 112, 219, 255),      # Purple - Ears (both)
            'chin': QColor(0, 255, 127, 255),       # Spring Green - Chin
            'cheek':  QColor(255, 105, 180, 255),   # Hot Pink - Cheeks
        }
        
        # Track occupied label regions to prevent overlap
        occupied_regions = []
        
        def draw_smart_label(x, y, text, category='zone', preferred_offset=(0, 0)):
            """Draw label with collision detection and auto-positioning"""
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            rect.adjust(-5, -2, 5, 2)
            
            # Try different positions to avoid overlap
            offsets_to_try = [
                preferred_offset,
                (0, 0),
                (0, -25),
                (0, 25),
                (30, 0),
                (-30, 0),
                (30, -25),
                (-30, -25),
                (30, 25),
                (-30, 25),
            ]
            
            final_rect = None
            for dx, dy in offsets_to_try:
                test_x, test_y = x + dx, y + dy
                test_rect = rect.translated(0, 0)
                test_rect.moveCenter(QPoint(test_x, test_y))
                
                # Check collision with existing labels
                collision = False
                for occupied in occupied_regions:
                    if test_rect.intersects(occupied):
                        collision = True
                        break
                
                if not collision:
                    final_rect = test_rect
                    break
            
            # If all positions collide, use preferred with larger vertical offset to force separation
            if final_rect is None:
                # Increase offset incrementally until no collision
                for multiplier in range(1, 10):
                    test_y = y + preferred_offset[1] + (25 * multiplier)
                    test_rect = rect.translated(0, 0)
                    test_rect.moveCenter(QPoint(x + preferred_offset[0], test_y))
                    
                    collision = False
                    for occupied in occupied_regions:
                        if test_rect.intersects(occupied):
                            collision = True
                            break
                    
                    if not collision:
                        final_rect = test_rect
                        break
                
                # Last resort: just use offset position
                if final_rect is None:
                    final_rect = rect.translated(0, 0)
                    final_rect.moveCenter(QPoint(x + preferred_offset[0], y + preferred_offset[1] + 50))
            
            # Get color for category
            bg_color = COLOR_PALETTE.get(category, QColor(100, 100, 100, 180))
            # Darker version for background
            bg_color_dark = QColor(bg_color.red()//2, bg_color.green()//2, bg_color.blue()//2, 180)
            
            # Draw background
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg_color_dark)
            painter.drawRoundedRect(final_rect, 4, 4)
            
            # Draw text
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(final_rect, Qt.AlignCenter, text)
            
            # Mark region as occupied
            occupied_regions.append(final_rect)
            
            return final_rect

        # --- 1. Facial Zones (Horizontal Lines) - RED ---
        
        cv_img = None
        if original_image is not None:
            cv_img = original_image
        else:
            # Convert pixmap to CV image for annotator (Safer QBuffer method)
            # NOTE: This is expensive and can be unstable. Prefer passing original_image.
            qimg = pixmap.toImage()
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            qimg.save(buffer, "PNG")
            data = np.frombuffer(buffer.data(), dtype=np.uint8)
            cv_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            
            if cv_img is None:
                 # Fallback
                 ptr = qimg.bits()
                 # Check format
                 fmt = qimg.format()
                 if fmt == QImage.Format_RGB888:
                     arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 3)
                     cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                 elif fmt == QImage.Format_RGBA8888:
                     arr = np.array(ptr).reshape(qimg.height(), qimg.width(), 4)
                     cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                 else:
                     print(f"DEBUG: Unsupported QImage format: {fmt}")
                     # Create black image as fallback
                     cv_img = np.zeros((h, w, 3), dtype=np.uint8)

        # Detect Hairline
        top_y = annotator.detect_hairline_y(cv_img, landmarks)
        
        brow_y = (p(105)[1] + p(334)[1]) // 2
        nose_y = p(2)[1]
        
        # Chin bottom with slight offset
        # Handle NormalizedLandmarkList for direct access
        lms = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        
        if len(lms) > 152:
            chin_lm = lms[152]
            chin_y = int(chin_lm.y * h) + int(h * 0.015)
        else:
            print("DEBUG: Landmarks too short for chin detection, using fallback.")
            chin_y = h - 10
            
        # Calculate Zone Lengths
        metrics = annotator.calculate_zone_metrics(cv_img, landmarks)
        z1_h, z1_w = metrics['z1_h'], metrics['z1_w']
        z2_h, z2_w = metrics['z2_h'], metrics['z2_w']
        z3_h, z3_w = metrics['z3_h'], metrics['z3_w']
        
        painter.setPen(pen_zone)
        
        # Draw horizontal lines across the face width (approx)
        left_x = p(234)[0]
        right_x = p(454)[0]
        margin = 40
        
        # Line 1: Top (Hairline)
        painter.drawLine(left_x - margin, top_y, right_x + margin, top_y)
        
        # Line 2: Brow
        painter.drawLine(left_x - margin, brow_y, right_x + margin, brow_y)
        
        # Line 3: Nose Bottom
        painter.drawLine(left_x - margin, nose_y, right_x + margin, nose_y)
        
        # Line 4: Chin
        painter.drawLine(left_x - margin, chin_y, right_x + margin, chin_y)
        
        # --- Draw Height Labels (Far Right) ---
        painter.setPen(COLOR_PALETTE['zone'])
        label_x = right_x + margin + 10
        draw_smart_label(label_x, (top_y + brow_y)//2, f"1. Boy: {z1_h}", 'zone', (0, 0))
        draw_smart_label(label_x, (brow_y + nose_y)//2, f"2. Boy: {z2_h}", 'zone', (0, 0))
        draw_smart_label(label_x, (nose_y + chin_y)//2, f"3. Boy: {z3_h}", 'zone', (0, 0))
        
        # --- Draw Width Lines (Vertical/Horizontal) - RED ---
        painter.setPen(QPen(QColor(255, 0, 0, 255), line_width, Qt.SolidLine)) # Solid red for width
        
        # Zone 1 Width: Forehead (70 - 300)
        z1_left = p(70); z1_right = p(300)
        z1_y = (top_y + brow_y) // 2 
        
        painter.drawLine(z1_left[0], z1_y - 10, z1_left[0], z1_y + 10) 
        painter.drawLine(z1_right[0], z1_y - 10, z1_right[0], z1_y + 10) 
        painter.drawLine(z1_left[0], z1_y, z1_right[0], z1_y) 
        draw_smart_label((z1_left[0] + z1_right[0])//2, z1_y - 15, f"1. En: {z1_w}", 'zone', (0, -15))
        
        # Zone 2 Width: Cheekbones (234 - 454)
        z2_left = p(234); z2_right = p(454)
        z2_y = (brow_y + nose_y) // 2
        
        painter.drawLine(z2_left[0], z2_y - 10, z2_left[0], z2_y + 10)
        painter.drawLine(z2_right[0], z2_y - 10, z2_right[0], z2_y + 10)
        painter.drawLine(z2_left[0], z2_y, z2_right[0], z2_y)
        draw_smart_label((z2_left[0] + z2_right[0])//2, z2_y - 15, f"2. En: {z2_w}", 'zone', (0, -15))
        
        # Zone 3 Width: Jaw (172 - 397)
        z3_left = p(172); z3_right = p(397)
        z3_y = (nose_y + chin_y) // 2
        
        painter.drawLine(z3_left[0], z3_y - 10, z3_left[0], z3_y + 10)
        painter.drawLine(z3_right[0], z3_y - 10, z3_right[0], z3_y + 10)
        painter.drawLine(z3_left[0], z3_y, z3_right[0], z3_y)
        draw_smart_label((z3_left[0] + z3_right[0])//2, z3_y - 15, f"3. En: {z3_w}", 'zone', (0, -15))
        
        # --- Draw Nose Measurements (Yellow) ---
        pen_nose = QPen(QColor(255, 215, 0, 255), line_width, Qt.SolidLine) # Gold/Yellow
        painter.setPen(pen_nose)
        
        # Landmarks
        y_133 = p(133)[1]; y_362 = p(362)[1]
        nasion_y = int((y_133 + y_362) / 2 - (pixmap.height() * 0.015))
        
        n_bot = p(2)
        alar_l = p(102); alar_r = p(331)
        
        # Length Line (Vertical) - Shifted slightly to avoid overlap with center
        mid_x = (p(168)[0] + n_bot[0]) // 2 
        painter.drawLine(mid_x, nasion_y, mid_x, n_bot[1])
        
        # Width Line (Horizontal)
        painter.drawLine(alar_l[0], alar_l[1], alar_r[0], alar_r[1])
        
        # Helper for Yellow Labels
        def draw_yellow_label(x, y, text):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            rect.adjust(-5, -2, 5, 2)
            rect.moveCenter(QPoint(x, y))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(180, 140, 0, 180)) 
            painter.drawRoundedRect(rect, 4, 4)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignCenter, text)
            
        # Labels
        painter.setPen(COLOR_PALETTE['nose'])
        n_len_px = abs(n_bot[1] - nasion_y)
        n_wid_px = abs(alar_r[0] - alar_l[0])
        
        draw_smart_label(mid_x + 10, (nasion_y + n_bot[1])//2, f"Burun Boy: {n_len_px}", 'nose', (10, 0))
        draw_smart_label((alar_l[0] + alar_r[0])//2, alar_l[1] + 35, f"Burun En: {n_wid_px}", 'nose', (0, 35))
        
        # --- Draw Eye Measurements (Cyan) ---
        pen_eye = QPen(QColor(0, 255, 255, 255), line_width, Qt.SolidLine) # Cyan
        painter.setPen(pen_eye)
        
        # Eye Widths
        l_eye_l = p(33); l_eye_r = p(133)
        r_eye_l = p(362); r_eye_r = p(263)
        
        painter.drawLine(l_eye_l[0], l_eye_l[1], l_eye_r[0], l_eye_r[1])
        painter.drawLine(r_eye_l[0], r_eye_l[1], r_eye_r[0], r_eye_r[1])
        
        # Inter-canthal Distance
        painter.drawLine(l_eye_r[0], l_eye_r[1], r_eye_l[0], r_eye_l[1])
        
        # Helper for Cyan Labels
        def draw_cyan_label(x, y, text):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            rect.adjust(-5, -2, 5, 2)
            rect.moveCenter(QPoint(x, y))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 100, 100, 180)) # Dark Cyan BG
            painter.drawRoundedRect(rect, 4, 4)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignCenter, text)
            
        painter.setPen(COLOR_PALETTE['eye'])
        l_wid = abs(l_eye_r[0] - l_eye_l[0])
        r_wid = abs(r_eye_r[0] - r_eye_l[0])
        icd = abs(r_eye_l[0] - l_eye_r[0])
        
        draw_smart_label((l_eye_l[0] + l_eye_r[0])//2, l_eye_l[1] - 20, f"Sol: {l_wid}", 'eye', (0, -20))
        draw_smart_label((r_eye_l[0] + r_eye_r[0])//2, r_eye_l[1] - 20, f"Sağ: {r_wid}", 'eye', (0, -20))
        draw_smart_label((l_eye_r[0] + r_eye_l[0])//2, l_eye_r[1] + 20, f"Ara: {icd}", 'eye', (0, 20))
        
        # --- Draw Lips Measurements (Magenta) ---
        pen_lips = QPen(QColor(255, 0, 255, 255), line_width, Qt.SolidLine) # Magenta
        painter.setPen(pen_lips)
        
        m_left = p(61); m_right = p(291)
        painter.drawLine(m_left[0], m_left[1], m_right[0], m_right[1])
        
        # Helper for Pink Labels
        def draw_pink_label(x, y, text):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            rect.adjust(-5, -2, 5, 2)
            rect.moveCenter(QPoint(x, y))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(100, 0, 100, 180)) # Dark Magenta BG
            painter.drawRoundedRect(rect, 4, 4)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignCenter, text)
            
        painter.setPen(COLOR_PALETTE['lip'])
        m_wid = abs(m_right[0] - m_left[0])
        draw_smart_label((m_left[0] + m_right[0])//2, m_left[1] + 30, f"Dudak En: {m_wid}", 'lip', (0, 30))
        
        # --- Draw Lip Thickness Measurements (Magenta - same as lip width) ---
        pen_lip_thickness = QPen(COLOR_PALETTE['lip'], line_width, Qt.SolidLine)
        painter.setPen(pen_lip_thickness)
        
        # Upper lip thickness - find thickest point
        # Upper lip outer edge landmarks: 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
        # Upper lip inner edge (lip line): 0, 37, 39, 40, 185, 267, 269, 270, 409
        upper_outer = [13, 312, 311, 310, 415, 409, 270, 269, 267, 185, 40, 39, 37, 0]
        upper_inner = [0, 267, 269, 270, 409, 415, 310, 311, 312, 13]
        
        max_upper_thickness = 0
        upper_thick_x = 0
        upper_top_y = 0
        upper_bot_y = 0
        
        # Find center x position for measurement
        center_x = p(0)[0]
        
        # Measure thickness along vertical line at center
        # Get upper lip vermilion border (outer edge)
        upper_vermilion = [p(13), p(312), p(311), p(310), p(415), p(308), p(324), p(318), p(402), p(317), p(14)]
        # Get upper lip line (cupid's bow)
        upper_line = [p(0), p(37), p(39), p(40), p(185), p(267), p(269), p(270), p(409)]
        
        # Find closest points to center
        closest_vermilion = min(upper_vermilion, key=lambda pt: abs(pt[0] - center_x))
        closest_line = min(upper_line, key=lambda pt: abs(pt[0] - center_x))
        
        upper_thickness = abs(closest_vermilion[1] - closest_line[1])
        upper_mid_x = (closest_vermilion[0] + closest_line[0]) // 2
        upper_top_y = min(closest_vermilion[1], closest_line[1])
        upper_bot_y = max(closest_vermilion[1], closest_line[1])
        
        # Draw upper lip thickness line
        painter.drawLine(upper_mid_x, upper_top_y, upper_mid_x, upper_bot_y)
        # Draw caps
        painter.drawLine(upper_mid_x - 5, upper_top_y, upper_mid_x + 5, upper_top_y)
        painter.drawLine(upper_mid_x - 5, upper_bot_y, upper_mid_x + 5, upper_bot_y)
        
        # Lower lip thickness - find thickest point
        # Lower lip SHOULD use landmark 17 (lower lip line), NOT landmark 0 (upper lip line)
        # Lower lip outer edge: 14, 317, 318, 402, 324, 308, 415, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61
        
        # Get lower lip line points (bottom edge of lip slit)
        lower_line_points = [p(17), p(84), p(181), p(91), p(146), p(61), p(291), p(375), p(321), p(405), p(314)]
        # Get lower lip vermilion border (outer bottom edge)
        lower_outer_points = [p(14), p(317), p(318), p(402), p(324), p(308), p(415), p(310), p(311), p(312), p(13)]
        
        # Find closest points to center for lower lip
        closest_lower_line = min(lower_line_points, key=lambda pt: abs(pt[0] - center_x))
        closest_lower_outer = min(lower_outer_points, key=lambda pt: abs(pt[0] - center_x))
        
        lower_thickness = abs(closest_lower_outer[1] - closest_lower_line[1])
        lower_mid_x = (closest_lower_outer[0] + closest_lower_line[0]) // 2
        lower_top_y = min(closest_lower_outer[1], closest_lower_line[1])
        lower_bot_y = max(closest_lower_outer[1], closest_lower_line[1])
        
        # Draw lower lip thickness line
        painter.drawLine(lower_mid_x, lower_top_y, lower_mid_x, lower_bot_y)
        # Draw caps
        painter.drawLine(lower_mid_x - 5, lower_top_y, lower_mid_x + 5, lower_top_y)
        painter.drawLine(lower_mid_x - 5, lower_bot_y, lower_mid_x + 5, lower_bot_y)
        
        # Labels for lip thickness - position next to the measurement line
        # Upper lip label: to the right of the measurement line, centered vertically
        draw_smart_label(upper_mid_x + 40, (upper_top_y + upper_bot_y)//2, f"Üst Dudak Boy: {upper_thickness}", 'lip', (40, 0))
        # Lower lip label: to the right of the measurement line, centered vertically
        draw_smart_label(lower_mid_x + 40, (lower_top_y + lower_bot_y)//2, f"Alt Dudak Boy: {lower_thickness}", 'lip', (40, 0))
        
        
        # --- Draw Ear Measurements (Purple) ---
        pen_ear = QPen(COLOR_PALETTE['ear'], line_width, Qt.SolidLine)
        painter.setPen(pen_ear)
        
        # Helper for Purple Labels
        def draw_purple_label(x, y, text):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text)
            rect.adjust(-5, -2, 5, 2)
            rect.moveCenter(QPoint(x, y))
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(75, 0, 130, 180)) # Indigo BG
            painter.drawRoundedRect(rect, 4, 4)
            
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(rect, Qt.AlignCenter, text)

        # Approximate Ear Height (Physiognomically between Brow and Nose base)
        # Right Ear (near 234)
        r_ear_x = p(234)[0] - 80 # Shift left significantly
        l_ear_x = p(454)[0] + 80 # Shift right significantly
        
        # Use dynamic detection with fallback and symmetry check
        r_conf = 0.0
        l_conf = 0.0
        
        if cv_img is not None:
            r_ear_top_y, r_ear_bot_y, r_conf = annotator.detect_ear_vertical_bounds(cv_img, landmarks, side='right')
            l_ear_top_y, l_ear_bot_y, l_conf = annotator.detect_ear_vertical_bounds(cv_img, landmarks, side='left')
            
            # Symmetry Logic: If one side is confident (>0.5) and other is not, sync them
            # Note: This assumes head is roughly upright.
            if r_conf > 0.5 and l_conf <= 0.5:
                # Right is better, sync Left to Right
                l_ear_top_y = r_ear_top_y
                l_ear_bot_y = r_ear_bot_y
            elif l_conf > 0.5 and r_conf <= 0.5:
                # Left is better, sync Right to Left
                r_ear_top_y = l_ear_top_y
                r_ear_bot_y = l_ear_bot_y
            
            # Sanity Check: If there is still a large discrepancy, trust the one closer to nose line
            # This handles cases where "bad" ear was detected with high confidence (false positive)
            if abs(r_ear_bot_y - l_ear_bot_y) > 30:
                nose_y = p(2)[1]
                r_dist = abs(r_ear_bot_y - nose_y)
                l_dist = abs(l_ear_bot_y - nose_y)
                
                if r_dist < l_dist:
                    # Right is anatomically more plausible
                    l_ear_top_y = r_ear_top_y
                    l_ear_bot_y = r_ear_bot_y
                else:
                    # Left is anatomically more plausible
                    r_ear_top_y = l_ear_top_y
                    r_ear_bot_y = l_ear_bot_y
        else:
            r_ear_top_y = p(127)[1] 
            r_ear_bot_y = p(93)[1]
            l_ear_top_y = p(356)[1]
            l_ear_bot_y = p(323)[1]
        
        # Draw Right Ear - RESET PEN COLOR
        painter.setPen(QPen(COLOR_PALETTE['ear'], line_width, Qt.SolidLine))
        painter.drawLine(r_ear_x, r_ear_top_y, r_ear_x, r_ear_bot_y)
        # Draw caps
        painter.drawLine(r_ear_x - 5, r_ear_top_y, r_ear_x + 5, r_ear_top_y)
        painter.drawLine(r_ear_x - 5, r_ear_bot_y, r_ear_x + 5, r_ear_bot_y)
        
        # Draw label BELOW the bottom cap to avoid overlap with side measurements
        draw_smart_label(r_ear_x, r_ear_bot_y + 20, f"Kulak Boy: {abs(r_ear_bot_y - r_ear_top_y)}", 'ear', (0, 20))
        
        # Draw Left Ear - RESET PEN COLOR
        painter.setPen(QPen(COLOR_PALETTE['ear'], line_width, Qt.SolidLine))
        painter.drawLine(l_ear_x, l_ear_top_y, l_ear_x, l_ear_bot_y)
        # Draw caps
        painter.drawLine(l_ear_x - 5, l_ear_top_y, l_ear_x + 5, l_ear_top_y)
        painter.drawLine(l_ear_x - 5, l_ear_bot_y, l_ear_x + 5, l_ear_bot_y)
        
        # Draw label BELOW the bottom cap
        draw_smart_label(l_ear_x, l_ear_bot_y + 20, f"Kulak Boy: {abs(l_ear_bot_y - l_ear_top_y)}", 'ear', (0, 20))
        
        # --- Additional Metrics Overlay (Bottom-Left) ---
        # Show ratios and measurements not already displayed on the image
        overlay_x = 20
        overlay_y = h - 140  # Start from bottom (increased height)
        overlay_line_height = 24
        
        # Get metrics (already calculated above)
        # metrics = annotator.calculate_zone_metrics(cv_img, landmarks)
        
        WLR = metrics.get('WLR', 0)
        jaw_angle = metrics.get('jaw_angle', 0)
        FW = metrics.get('FW', 0)
        JW = metrics.get('JW', 0)
        
        forehead_jaw_ratio = FW / JW if JW > 0 else 0
        
        # Detect Face Shape for display
        shape_info = annotator._detect_face_shape(metrics, landmarks, w, h)
        detected_shape = shape_info.get('shape', 'Belirsiz')
        
        # Draw overlay background
        overlay_bg = QColor(20, 20, 30, 220)
        painter.setPen(Qt.NoPen)
        painter.setBrush(overlay_bg)
        overlay_rect_height = overlay_line_height * 5 + 20
        painter.drawRoundedRect(overlay_x - 10, overlay_y - overlay_line_height - 10, 
                               260, overlay_rect_height, 8, 8)
        
        # Draw title
        painter.setFont(QFont("Arial", font_size, QFont.Bold))
        painter.setPen(QColor(137, 180, 250))  # Blue
        painter.drawText(overlay_x, overlay_y, "📊 Yüz Analizi")
        overlay_y += overlay_line_height + 5
        
        # Draw metrics
        painter.setFont(QFont("Arial", font_size - 2))
        painter.setPen(QColor(200, 200, 200))
        
        # 1. Face Shape
        painter.setPen(QColor(255, 215, 0)) # Gold for shape
        painter.drawText(overlay_x, overlay_y, f"• Şekil: {detected_shape}")
        painter.setPen(QColor(200, 200, 200)) # Reset
        overlay_y += overlay_line_height
        
        # 2. WLR
        painter.drawText(overlay_x, overlay_y, f"• En/Boy (WLR): {WLR:.2f}")
        overlay_y += overlay_line_height
        
        # 3. Jaw Angle
        painter.drawText(overlay_x, overlay_y, f"• Çene Açısı: {jaw_angle:.1f}°")
        overlay_y += overlay_line_height
        
        # 4. Forehead/Jaw
        painter.drawText(overlay_x, overlay_y, f"• Alın/Çene: {forehead_jaw_ratio:.2f}")
        
        painter.end()
        return pixmap

