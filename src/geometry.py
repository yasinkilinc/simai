import numpy as np

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle at p2 formed by p1-p2-p3 in degrees.
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def midpoint(p1, p2):
    """Calculates the midpoint between two 3D points."""
    return (p1 + p2) / 2
