# Excersice joint data


import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# left side
knee = [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value]
hip = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value]
ankle = [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
elbow = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
shoulder = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW]
wrist = [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX]

# joint data

squat_joint = [knee, hip, ankle]

pushup_joint = [elbow, shoulder, wrist, hip, knee]

left_armcurl_joint = [elbow]

lunge_joint = [knee, hip, ankle]

pullup_joint = [elbow, shoulder, wrist, hip, knee]
