import mediapipe as mp
import cv2
import numpy as np
import parameter as pm


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
# 특정 부위의 landmark 만 추출하는 함수

# 두 개의 포즈 랜드마크를 받아서 유클리드 거리를 계산하는 함수
# def calculate_pose_similarity(pose_landmarks1, pose_landmarks2): 
#     similarity_scores = []
#     for pose1, pose2 in zip(pose_landmarks1, pose_landmarks2):
#         pushup_idx = [pm.left_wrist[1], pm.left_elbow[1], pm.left_shoulder[1], pm.left_hip[1], pm.left_knee[1]]

#         pose1_array = np.array([(lm.x, lm.y) for lm in pose1.landmark[idx]: for idx in pushup_idx])
#         pose2_array = np.array([(lm.x, lm.y) for lm in pose2.landmark[idx]: for idx in pushup_idx])
#         similarity = np.linalg.norm(pose1_array - pose2_array) # 
#         similarity_scores.append(similarity)

#     return similarity_scores

def calculate_pose_similarity(pose_landmarks1, pose_landmarks2):
    similarity_scores = []  # 동작 유사도를 저장하는 리스트
    for pose1, pose2 in zip(pose_landmarks1, pose_landmarks2):
        # 유크리드 거리 초기화
        sum = 0

        # 운동 동작의 랜드마크 인덱스 추출
        for idx in pm.joint['Pushup']:
            mov1 = pose1.landmark[idx[1]]
            mov2 = pose2.landmark[idx[1]]
            sum += np.linalg.norm(np.array([mov1.x, mov1.y]) - np.array([mov2.x, mov2.y]))
        
        # 유사도 평균 계산
        similarity = 1 - sum / len(pm.joint['Pushup'])
        similarity_scores.append(similarity)

    return similarity_scores

if __name__ == '__main__':
    # 두 개의 동영상 파일 경로
    video_path1 = 'ML/feedback/Dataset/score/video1.mp4'
    video_path2 = 'ML/feedback/Dataset/score/video2.mp4'

    # 포즈 추출
    pose_landmarks1 = get_pose_landmarks(video_path1)
    pose_landmarks2 = get_pose_landmarks(video_path2)

    # 유사도 계산
    similarity_scores = calculate_pose_similarity(pose_landmarks1, pose_landmarks2)

    # 유사도 출력
    for i, similarity in enumerate(similarity_scores):
        print(f"Frame {i+1}: {similarity}")
# similarity_scores가 0에 가까울수록 두 포즈가 비슷하다는 의미입니다.