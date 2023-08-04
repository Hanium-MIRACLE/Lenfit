import math
import numpy as np

def calculate_2d(landmark1,landmark2,landmark3):
    
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

    if angle < 0:
        angle += 360
    
    return angle

def calculate_3d(landmark1,landmark2,landmark3):
    
    landmark_1_2 = [(landmark2[i] - landmark1[i]) for i in range(3)]
    landmark_2_3 = [(landmark2[i] - landmark3[i]) for i in range(3)]

    dot_product = sum([landmark_1_2[i] * landmark_2_3[i] for i in range(3)])
    
    landmark_1_2_mag = np.sqrt(sum([coord**2 for coord in landmark_1_2]))
    landmark_2_3_mag = np.sqrt(sum([coord**2 for coord in landmark_2_3]))

    angle = np.degrees(np.arccos(dot_product / (landmark_1_2_mag * landmark_2_3_mag)))

    return angle
    