import numpy as np
import copy 
import itertools
import os
import matplotlib.pyplot as plt
import cv2
import sys
import tqdm
import time
import json
import datetime

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from src.func import *
from src.srcs import *
from src.score import Score

# Counting Repetitions
class AnalysisTempoCount:
    def __init__(self, fitness, video_path, out_video_dir, sample_csv_path, mode = 'Video', show = True):
        
        assert fitness in ['pushups', 'squat'], 'Unexpected fitness: {}'.format(fitness)
        assert mode in ['Video', 'Webcam'], 'Unexpected mode: {}'.format(mode)
        
        # 각 운동별 주요 관절 정보
        if fitness == 'squat':
          self.main_land_name = {'left' : ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle'], 'right' : ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle']}
          self.main_land_num = {'left' : [11, 23, 25, 27], 'right' : [12, 24, 26, 28]}
        if fitness == 'pushups':
          self.main_land_name = {'left' : ['left_wrist', 'left_elbow', 'left_shoulder', 'left_hip', 'left_knee'], 'right' : ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']}
          self.main_land_num = {'left' : [15, 13, 11, 23, 25], 'right' : [16, 14, 12, 24, 26]}
          
    
        self.land_name = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
                              'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
                              'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 
                              'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 
                              'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
        
        # 스쿼트의 경우, 어깨, 엉덩이, 무릎, 발목 관절을 제외한 나머지 관절들은 제외한다. 
        if fitness == 'squat':
            self.exclude_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19 ,20, 21, 22]
        else :
            self.exclude_landmarks = []
        
        # Path to video file.
        self.video_path = video_path
        self.fitness = fitness
        self.out_video_path = out_video_dir
        self.class_name = f'{self.fitness}_down'
        self.pose_samples_folder = os.path.join(sample_csv_path, fitness)
        
        # Video parameters.
        self.video = cv2.VideoCapture(self.video_path if mode == 'Video' else 0)
        
        self.video_n_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        
        """
        Pose estimation에 필요한 객체들을 초기화
        """
        
        # Initialize tracker.
        self.pose_tracker = mp_pose.Pose()
        # Initialize embedder.
        self.pose_embedder = FullBodyPoseEmbedder(self.fitness)
        # Initialize classifier.
        self.pose_classifier = PoseClassifier(
                                    pose_samples_folder=self.pose_samples_folder,
                                    pose_embedder=self.pose_embedder,
                                    top_n_by_max_distance=30,
                                    top_n_by_mean_distance=10,
                                    n_landmarks=12 if self.fitness == 'squat' else 33)
        
        # Initialize EMA smoothing.
        self.pose_classification_filter = EMADictSmoothing(
                                        window_size=10,
                                        alpha=0.2)

        # Initialize counter.
        self.repetition_counter = RepetitionCounter(
            class_name=self.class_name,
            enter_threshold=9,
            exit_threshold=1)
        
        # Initialize renderer.
        self.pose_classification_visualizer = PoseClassificationVisualizer(
            class_name=self.class_name,
            plot_x_max=self.video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10)
        
        # 점수 계산 클래스 초기화
        self.get_score_by_angle = Score(target_dir= "data/angle_info", 
                                        out_plot_dir= out_video_dir,
                                        fitness= self.fitness)

        
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()
            
        # 실사용은 Webcam 모드 사용
        if mode == 'Webcam':
            self.Count_analysis_tempo(show)
            
        # 개발을 위해 Video모드 사용
        else:
            self.analysis_tempo(show)
        
        self.get_score_by_angle.draw_save_angle_plot()   

    def analysis_tempo(self, show=False):
        
        self.frame_idx = 0
        output_frame = None
        
        total_landmark_dict = {}
        total_landmark_dict['value'] = []
        total_landmark_dict['total_frame'] = 0
        
        n_out_video = 0
        one_tempo_angle = copy.deepcopy(self.get_score_by_angle.angle_info)
        one_tempo_angle_tmp = copy.deepcopy(self.get_score_by_angle.angle_info)
            
        out_video_name = f'{self.fitness}_{n_out_video}.mp4'
        out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))

        with tqdm.tqdm(total=self.video_n_frames, position=0, leave=True) as pbar:
            while True:
                # Get next frame of the video.
                success, input_frame = self.video.read()
                if not success:
                    break

                # Run pose tracker.
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                result = self.pose_tracker.process(image=input_frame)
                pose_landmarks = result.pose_landmarks

                # Draw pose prediction.
                output_frame = input_frame.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=output_frame,
                        landmark_list=pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)

                if pose_landmarks is not None:
                    
                    pose_landmarks_dict = {}
                    for idx, lmk in enumerate(pose_landmarks.landmark):
                        pose_landmarks_dict[self.land_name[idx]] = [lmk.x, lmk.y, lmk.z, lmk.visibility]
                        
                    
                    total_landmark_dict['value'].append(pose_landmarks_dict)
                    
                # Get landmarks.
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    # pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                    #                             for lmk in pose_landmarks.landmark], dtype=np.float32)
                    
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                for idx, lmk in enumerate(pose_landmarks.landmark) if idx not in self.exclude_landmarks],
                                                dtype=np.float32)
                    assert pose_landmarks.shape == ((12, 3) if self.fitness == 'squat' else (33, 3)), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                    # Classify the pose on the current frame.
                    pose_classification = self.pose_classifier(pose_landmarks)

                    # Smooth classification using EMA.
                    pose_classification_filtered = self.pose_classification_filter(pose_classification)

                    # Count repetitions.
                    repetitions_count = self.repetition_counter(pose_classification)
                else:
                # No pose => no classification on current frame.
                    pose_classification = None

                    # Still add empty classification to the filter to maintaing correct
                    # smoothing for future frames.
                    pose_classification_filtered = self.pose_classification_filter(dict())

                    # Don't update the counter presuming that person is 'frozen'. Just
                    # take the latest repetitions count.
                    repetitions_count = self.repetition_counter.n_repeats
                    
                # 각도 변화를 계산하고 저장하기 위해 Score 클래스의 __call__ 함수 호출
                one_tempo_angle = self.get_score_by_angle(pose_landmarks=pose_landmarks, 
                                                              one_tempo_angle=one_tempo_angle)           
                    
                # Up 과 Down의 시작점을 구분하기 위해
                start_pose = f"{self.fitness}_up" if self.class_name == f"{self.fitness}_down" else f"{self.fitness}_down"
                
                # Classification 한 결과를 통해 Count를 한다.
                if repetitions_count > n_out_video:
                    if start_pose in pose_classification_filtered and pose_classification_filtered[start_pose] == 9.999999999999998:
                    
                        n_out_video += 1
                        for side in ['left', 'right']:
                            for i in range(1, len(self.main_land_name[side]) - 1):
                                # 한 Tempo에 저장된 각도 변화 정보를 Score 클래스의 angle_info에 저장 -> {'left_shoulder' : [[첫번째 횟수 각도 변화], [두번째 횟수 각도 변화], ...], ...}
                                self.get_score_by_angle.angle_info[side][self.main_land_name[side][i]].append(one_tempo_angle[side][self.main_land_name[side][i]])
                                
                        del(one_tempo_angle)
                        one_tempo_angle = copy.deepcopy(one_tempo_angle_tmp)
                        self.last_frame_idx = self.frame_idx + 1
                        out_video.release()
                        out_video_name = f'{self.fitness}_{n_out_video}.mp4'
                        out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))
                
                
                # Draw classification plot and repetition counter.
                output_frame = self.pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered = pose_classification_filtered,
                    repetitions_count=repetitions_count)

                # Draw Classification Result on the frame.
                
                output_frame = np.array(output_frame)
                cv2.putText(output_frame, f"{pose_classification_filtered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Save the output frame.
                
                out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                if show:
                    cv2.imshow('Output Video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                    # Exit the loop if 'q' is pressed.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_idx += 1
                pbar.update()

        self.last_frame_idx = self.frame_idx
        
        total_landmark_dict['total_frame'] = self.frame_idx
        
        
        
        # Save and Close Video
        out_video.release()
        self.video.release()
        
        self.dict_to_json(total_landmark_dict, self.out_video_path)
        
        print("Save video path :", self.out_video_path)
        
        # Release MediaPipe resources.
        self.pose_tracker.close()


    def Count_analysis_tempo(self, show=False):
        """
        
        실시간 카메라를 사용하여 운동 Counting하는 함수.
        
        10초 카운트다운을 화면에 표시한 후 운동을 시작한다.
        
        이하 analysis_tempo와 동일하다.
        
        """
        self.frame_idx = 0
        output_frame = None
        
        n_out_video = 0
        one_tempo_angle = copy.deepcopy(self.angle_info)
        one_tempo_angle_tmp = copy.deepcopy(self.angle_info)
            
        out_video_name = f'{self.fitness}_{n_out_video}.mp4'
        out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))

        # Counting
        start_time = time.time()
        countdown_over = False

        with tqdm.tqdm(total=self.video_n_frames, position=0, leave=True) as pbar:
            while True:
                # Get next frame of the video.
                success, input_frame = self.video.read()
                if not success:
                    break
                
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                
                # 10초 카운트다운
                if not countdown_over:
                    elapsed_time = int(time.time() - start_time)
                    remaining_time = max(10 - elapsed_time, 0)

                    output_frame = input_frame.copy()
                    frame = cv2.putText(
                        img=output_frame,
                        text=f"Countdown: {remaining_time}",
                        org=(50, 200),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=(255, 255, 255),
                        thickness=5
                    )

                    if remaining_time <= 0:
                        countdown_over = True
    
                # 카운트다운이 끝나면 운동 시작
                if countdown_over:
                    # Run pose tracker.
                    result = self.pose_tracker.process(image=input_frame)
                    pose_landmarks = result.pose_landmarks

                    # Draw pose prediction.
                    output_frame = input_frame.copy()
                    if pose_landmarks is not None:
                        mp_drawing.draw_landmarks(
                            image=output_frame,
                            landmark_list=pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)

                    if pose_landmarks is not None:
                    # Get landmarks.
                        frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                        # pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                        #                             for lmk in pose_landmarks.landmark], dtype=np.float32)
                        
                        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                    for idx, lmk in enumerate(pose_landmarks.landmark) if idx not in self.exclude_landmarks],
                                                    dtype=np.float32)
                        assert pose_landmarks.shape == ((12, 3) if self.fitness == 'squat' else (33, 3)), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                        # Classify the pose on the current frame.
                        pose_classification = self.pose_classifier(pose_landmarks)

                        # Smooth classification using EMA.
                        pose_classification_filtered = self.pose_classification_filter(pose_classification)

                        # Count repetitions.
                        repetitions_count = self.repetition_counter(pose_classification)
                    else:
                    # No pose => no classification on current frame.
                        pose_classification = None

                        # Still add empty classification to the filter to maintaing correct
                        # smoothing for future frames.
                        pose_classification_filtered = self.pose_classification_filter(dict())

                        # Don't update the counter presuming that person is 'frozen'. Just
                        # take the latest repetitions count.
                        repetitions_count = self.repetition_counter.n_repeats
                        
                    # Calculate angle of the main landmarks and Save the angle info
                    one_tempo_angle = self.get_score_by_angle(pose_landmarks=pose_landmarks, 
                                                              one_tempo_angle=one_tempo_angle)
                        
                    start_pose = f"{self.fitness}_up" if self.class_name == f"{self.fitness}_down" else f"{self.fitness}_down"
                        
                    if repetitions_count > n_out_video:
                        if start_pose in pose_classification_filtered and pose_classification_filtered[start_pose] == 9.999999999999998:
                        
                            n_out_video += 1
                            for side in ['left', 'right']:
                                for i in range(1, len(self.main_land_name[side]) - 1):
                                    self.get_score_by_angle.angle_info[side][self.main_land_name[side][i]].append(one_tempo_angle[side][self.main_land_name[side][i]])
                                    
                            del(one_tempo_angle)
                            one_tempo_angle = copy.deepcopy(one_tempo_angle_tmp)
                            self.last_frame_idx = self.frame_idx + 1
                            out_video.release()
                            out_video_name = f'{self.fitness}_{n_out_video}.mp4'
                            out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))
                    
                    
                    # Draw classification plot and repetition counter.
                    output_frame = self.pose_classification_visualizer(
                        frame=output_frame,
                        pose_classification=pose_classification,
                        pose_classification_filtered = pose_classification_filtered,
                        repetitions_count=repetitions_count)

                    # Draw Classification Result on the frame.
                    
                    output_frame = np.array(output_frame)
                    cv2.putText(output_frame, f"{pose_classification_filtered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Save the output frame.
                    
                    out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                    if show:
                        cv2.imshow('Output Video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                        # Exit the loop if 'q' is pressed.
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    self.frame_idx += 1
                    pbar.update()

        self.last_frame_idx = self.frame_idx
        
        # Save and Close Video
        out_video.release()
        self.video.release()
        

        
        print("Save video path :", self.out_video_path)
        
        # Release MediaPipe resources.
        self.pose_tracker.close()
    
    def get_angle_info(self):
      return self.get_score_by_angle.angle_info
  
  
    # dict to json
    def dict_to_json(self, data, output_dir):
        json_pth = os.path.join(output_dir, 'test.json')
        with open(json_pth, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4)

class Person:
    def __init__(self, name, preferred_fitness, input_video_dir = None, sample_csv_path = 'data/fitness_poses_csvs_out'):
        self.name = name
        self.preferred_fitness = preferred_fitness
        self.last_fitness = None
        self.input_video_dir = input_video_dir
        self.sample_csv_path = sample_csv_path
        self.save_video_dir = f'Result/{name}'
        
        self.fitness_angle_info = {}
        
        self.score = {}
        
        try:
            # 해당 사람의 운동별 결과를 저장할 디렉토리 생성
            if not os.path.exists(self.save_video_dir):
                os.makedirs(self.save_video_dir)
                
            # 사람 디렉토리 안에 현재 날짜로 된 디렉토리 생성
            if not os.path.exists(os.path.join(self.save_video_dir, self.get_today())):
                os.makedirs(os.path.join(self.save_video_dir, self.get_today()))
            
            # 결과 저장할 디렉토리 Update
            self.save_video_dir = os.path.join(self.save_video_dir, self.get_today())
            
            # 운동별 디렉토리 생성
            for fitness in preferred_fitness:
                path = os.path.join(self.save_video_dir, fitness)
                if not os.path.exists(path):
                    os.makedirs(path)
                    
            
        except:
            print("Error to make directory")

    def get_today(self):
        now = datetime.datetime.now()
        date = now.strftime('%Y-%m-%d')

        return date
        
    def get_name(self):
        return self.name
    
    def get_now_time(self):
        return datetime.datetime.now()
    
    def get_preferred_fitness(self):
        return self.preferred_fitness
    
    def get_last_fitness(self):
        return self.last_fitness

    def get_fitness_angle_info(self):
        return self.fitness_angle_info
    
    def get_score(self):
        return self.score
    
    def dict_to_json(self, data, fitness, output_dir):

        json_pth = os.path.join(output_dir, f'{fitness}Score.json')
        with open(json_pth, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4)
  
    def analyze_fitness(self, fitness, mode = 'Webcam'):
        self.last_fitness = fitness
        print(f"----------{self.name} starts {fitness}----------")
        
        input_video = f'{self.input_video_dir}/{fitness}.mp4'
        output_dir = os.path.join(self.save_video_dir, fitness)
        
        # 운동 Counting, 객체 호출과 동시에 Counting 시작
        fitness_analyzer = AnalysisTempoCount(fitness = fitness, 
                                            mode = mode,
                                            show = True, 
                                            video_path = input_video,
                                            out_video_dir = output_dir, 
                                            sample_csv_path = self.sample_csv_path)

        self.fitness_angle_info[fitness] = fitness_analyzer.get_angle_info()
        
        # 각도 변화 정보를 통해 점수 계산
        score = fitness_analyzer.get_score_by_angle.calculate_score(self.fitness_angle_info[fitness])
        
        self.score[fitness] = score
        
        # 점수를 json 파일로 저장
        self.dict_to_json(score, fitness, output_dir)
        
        print(f"----------{self.name} ends {fitness}----------")
  
  