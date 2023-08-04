import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import function as fn

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dense



actions = ['barbell', 'body']
seq_length = 30

model = Sequential([
    LSTM(64, activation='relu', input_shape=(30,160)),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.load_weights('/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/squat_weight')


# MediaPipe pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

file_body = '/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset/bodySquat/bodysquat02.mp4'
file_barbell = '/Users/jiwonryu/WorkSpace/lenfitDemo2/Lenfit-main/ML/pose_classification/dataset/barbellSquat/barbellsquat02.mp4'

cap = cv2.VideoCapture(file_body)

First = [12, 12, 13, 11, 13, 13, 13, 21, 11, 11, 24, 23, 25, 25,
         11, 11, 14, 12, 14, 12, 14, 22, 12, 12, 23, 24, 26, 26]
Mid =   [11, 11, 11, 13, 15, 15, 15, 15, 23, 23, 23, 25, 27, 27,
         12, 12, 12, 14, 16, 14, 16, 16, 24, 14, 24, 26, 28, 28]
End =   [13, 23, 23, 15, 21, 19, 17, 19, 24, 25, 25, 27, 31, 29,
         14, 24, 24, 16, 22, 20, 18, 20, 23, 26, 26, 28, 32, 30]

seq = []
action_seq = []

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
    

        d = np.concatenate([joint.flatten(), angle])

        seq.append(d)

        
        if len(seq) < seq_length:
                continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

    # cv2.imshow('img', image)
    if cv2.waitKey(1) == ord('q'):
        break

print('fitness:',this_action)