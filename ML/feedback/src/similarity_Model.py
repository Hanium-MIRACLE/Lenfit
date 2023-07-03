import mediapipe as mp
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_pose_landmarks(video_path):
    mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)

    pose_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(frame)
        if results.pose_landmarks:
            pose_landmarks.append(results.pose_landmarks)

    cap.release()
    mp_pose.close()
    return pose_landmarks

def calculate_pose_similarity(pose_landmarks1, pose_landmarks2):
    similarity_scores = []  # 동작 유사도를 저장하는 리스트
    for pose1, pose2 in zip(pose_landmarks1, pose_landmarks2):
        # 특정 랜드마크의 인덱스 추출
        shoulder_idx = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
        hip_idx = mp.solutions.pose.PoseLandmark.LEFT_HIP.value

        # pose1과 pose2에서 어깨와 엉덩이의 좌표 추출
        shoulder1 = pose1.landmark[shoulder_idx]
        hip1 = pose1.landmark[hip_idx]
        shoulder2 = pose2.landmark[shoulder_idx]
        hip2 = pose2.landmark[hip_idx]

        # 어깨와 엉덩이 간의 유클리디안 거리 계산
        distance = np.linalg.norm(np.array([shoulder1.x, shoulder1.y]) - np.array([shoulder2.x, shoulder2.y])) \
                   + np.linalg.norm(np.array([hip1.x, hip1.y]) - np.array([hip2.x, hip2.y]))

        similarity_scores.append(distance)

    return similarity_scores

if __name__ == '__main__':
    # 두 개의 동영상 파일 경로
    video_path1 = 'ML/feedback/Dataset/score/video1.mp4'
    video_path2 = 'ML/feedback/Dataset/score/video2.mp4'

    # 포즈 추출
    pose_landmarks1 = get_pose_landmarks(video_path1)
    pose_landmarks2 = get_pose_landmarks(video_path2)

    # 동작 유사도 계산
    similarity_scores = calculate_pose_similarity(pose_landmarks1, pose_landmarks2)

    # 데이터셋 생성
    X = np.array(similarity_scores).reshape(-1, 1)
    y = np.array([1, 0])  # 예시로 두 동영상이 동일한 동작이면 1, 다른 동작이면 0으로 라벨링한 것입니다.

    # 데이터셋 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 신경망 모델 구성
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 모델 학습
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
