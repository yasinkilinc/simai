import cv2
import os

class InputProcessor:
    def __init__(self, source=0):
        """
        Initialize InputProcessor.
        :param source: Path to video file, image file, or camera index (default 0).
        """
        self.source = source
        self.cap = None
        self.is_image = False

    def load_source(self):
        """
        Loads the video or image source.
        """
        if isinstance(self.source, str) and (self.source.lower().endswith(('.png', '.jpg', '.jpeg'))):
            self.is_image = True
            self.frame = cv2.imread(self.source)
            if self.frame is None:
                raise ValueError(f"Could not load image from {self.source}")
        else:
            self.is_image = False
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source {self.source}")

    def get_frame(self):
        """
        Yields frames from the source.
        """
        if self.is_image:
            yield self.frame
        else:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
            self.cap.release()

    def release(self):
        if self.cap:
            self.cap.release()
