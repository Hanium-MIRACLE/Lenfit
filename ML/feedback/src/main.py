import numpy as np
import mediapipe as mp 
import cv2

import parameter as pm
import function as fn


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

state = pm.workout[3] # 0: Squat, 1: Pushup, 2: Lunge, 3: Pullup, 4: Armcurl

cap = cv2.VideoCapture(pm.adress[state])

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isls
    
        # read video
        ret, frame = cap.read()
        if not ret: break # if video is end, break

        # frame size
        height, width, _ = frame.shape

        # color change
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        # detection
        results = pose.process(img)

        # draw landmarks
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # exception 
        # try:
        landmarks = results.pose_landmarks.landmark
        angle_dict = dict()

        # ------------------- feedback ------------------- #
        for i, jnts in enumerate(pm.joint[state]): 
            # get coordinates
            xy_data = [[landmarks[jnt].x, landmarks[jnt].y] for jnt in jnts]

            # calculate angle
            angle = np.abs(90 - fn.calculate_angle(xy_data[0], xy_data[1], xy_data[1]+[0, -1])) # vertical angel
            angle_dict[jnts[1]] = angle
            # feedback
            comment = fn.feedback(angle, pm.limit[state][i], 'example') # pm.comment[state][i] 추가

            # visualize angle
            cv2.putText(img, str(angle) +'    '+ str(comment) ,
                        tuple(np.multiply(xy_data[1], [width, height]).astype(int)), # np.multiply(xy_data[1], [640, 480]).astype(int)의 의미: x, y 좌표를 640, 480으로 곱해준다.
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                        )
            # cv2.putText(img, str(fb), (1000, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # except:
        #     print('error')
        # ------------------- feedback ------------------- #


        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )
        
        cv2.imshow('VideoFrame', img)
        if cv2.waitKey(1) == ord('q'):
            break



# 동영상 재생 종료
cap.release()
# 화면에 나타난 윈도우들을 종료
cv2.destroyAllWindows()
