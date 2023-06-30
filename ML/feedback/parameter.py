# Excersice joint data

import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# left side
left_knee = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]
left_hip = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value]
left_ankle = [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
left_elbow = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
left_shoulder = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW]
left_wrist = [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX]

# right side
right_knee = [mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value]
right_hip = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]
right_ankle = [mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
right_elbow = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
right_shoulder = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW]
right_wrist = [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX]

# ------------------- workout -------------------
workout = ['Squat', 'Pushup', 'Lunge', 'Pullup', 'ArmCurl']

# ------------------- video -------------------
adress = {'Squat':'/Users/iyongbin/Repository/Lenfit/ML/Dataset/video/Squat.mp4',
          'Pushup':'/Users/iyongbin/Repository/Lenfit/ML/Dataset/video/Pushup.mp4',
          'Lunge':'/Users/iyongbin/Repository/Lenfit/ML/Dataset/video/Lunge.mp4',
          'Pullup':'/Users/iyongbin/Repository/Lenfit/ML/Dataset/video/Pullup.mp4',
          'ArmCurl':'/Users/iyongbin/Repository/Lenfit/ML/Dataset/video/ArmCurl.mp4'
          }

# ------------------- joint -------------------

joint = {'Squat': [left_knee, left_hip, left_ankle], # 3개
        'Pushup': [left_wrist, left_elbow, left_shoulder, left_hip, left_knee], # 5개
        'Lunge': [left_knee, left_hip, left_ankle, right_knee, right_hip, right_ankle], # 6개
        'Pullup': [left_wrist, left_elbow, left_shoulder, right_wrist, right_elbow, right_shoulder], # 6개
        'ArmCurl': [left_elbow, right_elbow] # 2개
         }

# ------------------- angle -------------------
limit = {
    'Squat': [84, 27, 28],
    'Pushup': [0, 0, 0, 0, 0],
    'Lunge': [0, 0, 0, 0, 0, 0],
    'Pullup': [0, 0, 0, 0, 0, 0],
    'ArmCurl': [0, 0]
}

# ------------------- comment -------------------
comment = {
    'Squat': ['Knee is too much bent', 'Hip is too much bent', 'Ankle is too much bent'],
    'Pushup': ['elbow is too much bent', 'shoulder is too much bent', 'wrist is too much bent', 'hip is too much bent', 'knee is too much bent'],
    'Lunge': ['Knee is too much bent', 'Hip is too much bent', 'Ankle is too much bent'],
    'Pullup': ['elbow is too much bent', 'shoulder is too much bent', 'wrist is too much bent', 'hip is too much bent', 'knee is too much bent'],
    'ArmCurl': ['elbow is too much bent']
}

# ------------------- init -------------------
if __name__ == '__main__':
    for i in range(len(workout)):
        print(workout[i])
        print(len(joint[workout[i]]))
        print(len(limit[workout[i]]))
        print(len(comment[workout[i]]))
        print('-------------------')