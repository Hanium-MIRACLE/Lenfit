import cv2
import mediapipe as mp
import numpy as np
import function as fn
import time, os
import math

actions = ['barbell', 'body']
seq_length = 30
secs_for_action = 15

created_time = int(time.time())

# landmark index
First = [12, 12, 13, 11, 13, 13, 13, 21, 11, 11, 24, 23, 25, 25,
         11, 11, 14, 12, 14, 12, 14, 22, 12, 12, 23, 24, 26, 26]
Mid =   [11, 11, 11, 13, 15, 15, 15, 15, 23, 23, 23, 25, 27, 27,
         12, 12, 12, 14, 16, 14, 16, 16, 24, 14, 24, 26, 28, 28]
End =   [13, 23, 23, 15, 21, 19, 17, 19, 24, 25, 25, 27, 31, 29,
         14, 24, 24, 16, 22, 20, 18, 20, 23, 26, 26, 28, 32, 30]


# MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

file_body = '/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset/bodySquat/bodysquat05.mp4'
file_barbell = '/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset/barbellSquat/barbellsquat08.mp4'

data = []
label = 'body'
cap = cv2.VideoCapture(file_body)
print(f'* * * Start creating {label} dataset * * * ')

while cap.isOpened():

    ret, image = cap.read()

    if not ret:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    result = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks is not None:
        res = result.pose_landmarks.landmark
        joint = np.zeros((33, 4))
        for idx in range(33):
            joint[idx] = [res[idx].x, res[idx].y, res[idx].z, res[idx].visibility]
        
        angle = np.zeros((len(First),))

        for idx in range(len(angle)):
            angle[idx] = fn.calculate_3d(joint[First[idx], :3],joint[Mid[idx], :3],joint[End[idx], :3])
        

        angle_label = np.array([angle], dtype=np.float32)
        angle_label = np.append(angle_label, label)

        d = np.concatenate([joint.flatten(), angle_label])

        data.append(d)

    if cv2.waitKey(5) & 0xFF == 27:
        break

data = np.array(data)
print(label, data.shape)
np.save(os.path.join('/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset', f'raw_{label}_{created_time}'), data)

# Create sequence data
full_seq_data = []
for seq in range(len(data) - seq_length):
    full_seq_data.append(data[seq:seq + seq_length])

full_seq_data = np.array(full_seq_data)
print(label, full_seq_data.shape)
np.save(os.path.join('/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset', f'seq_{label}_{created_time}'), full_seq_data)

data = []
label = 'barbell'
cap = cv2.VideoCapture(file_barbell)
print(f'* * * Start creating {label} dataset * * * ')

while cap.isOpened():

    ret, image = cap.read()
    if not ret:
        break


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    
    result = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks is not None:
        res = result.pose_landmarks.landmark
        joint = np.zeros((33, 4))
        for idx in range(33):
            joint[idx] = [res[idx].x, res[idx].y, res[idx].z, res[idx].visibility]
        
        angle = np.zeros((len(First),))

        for idx in range(len(angle)):
            angle[idx] = fn.calculate_3d(joint[First[idx], :3],joint[Mid[idx], :3],joint[End[idx], :3])
        

        angle_label = np.array([angle], dtype=np.float32)
        angle_label = np.append(angle_label, label)

        d = np.concatenate([joint.flatten(), angle_label])

        data.append(d)

    if cv2.waitKey(5) & 0xFF == 27:
        break

data = np.array(data)
print(label, data.shape)
np.save(os.path.join('/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset', f'raw_{label}_{created_time}'), data)

# Create sequence data
full_seq_data = []
for seq in range(len(data) - seq_length):
    full_seq_data.append(data[seq:seq + seq_length])

full_seq_data = np.array(full_seq_data)
print(label, full_seq_data.shape)
np.save(os.path.join('/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset', f'seq_{label}_{created_time}'), full_seq_data)
