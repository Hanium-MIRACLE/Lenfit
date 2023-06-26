# mediapipe pose estimation 운동 자세 인식
# 2021.07.08

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

# 1. Pose Landmark
# 2. Pose Connection
# 3. Pose Estimation
# 4. Pose Tracking
# 5. Pose Angle
# 6. Pose Angle Detection
# 7. Pose Angle Tracking
# 8. Pose Angle Detection and Tracking
# 9. Pose Angle Detection and Tracking with OpenCV
# 10. Pose Angle Detection and Tracking with OpenCV and Mediapipe
# 11. Pose Angle Detection and Tracking with OpenCV and Mediapipe - 2
# 12. Pose Angle Detection and Tracking with OpenCV and Mediapipe - 3



