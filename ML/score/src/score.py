import numpy as np
import pandas as pd
import copy 
import itertools
import os
import matplotlib.pyplot as plt
import sys
import time
import json

from scipy.spatial.distance import cosine

from src.srcs import *

class TempoScore:
    def __init__(self, fitness, save_result_dir, master_angle_dir='./util/sample_angle', sample_csv_path='./util/sample_pose'):
        
        assert fitness in ['pushups', 'squat'], 'Unexpected fitness: {}'.format(fitness)
        
        # 각 운동별 주요 관절 정보
        if fitness == 'squat':
            self.main_land_name = {'left' : ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'], 'right' : ['RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE']}
          
        if fitness == 'pushups':
          
            self.main_land_name = {'left' : ['LEFT_WRIST', 'LEFT_ELBOW', 'LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_KNEE'], 'right' : ['RIGHT_WRIST', 'RIGHT_ELBOW', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE']}

        # 스쿼트의 경우, 어깨, 엉덩이, 무릎, 발목 관절을 제외한 나머지 관절들은 제외한다. 
        if fitness == 'squat':
            self.land_name = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE','LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
            self.exclude_landmarks = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
                                      'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
                                      'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
                                      'LEFT_THUMB', 'RIGHT_THUMB']
        else :
            self.land_name = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 
                              'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
                              'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 
                              'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 
                              'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
            self.exclude_landmarks = []
        
        self.fitness = fitness
        self.class_name = f'{self.fitness}_down' if self.fitness == 'squat' or self.fitness == 'pushups' else f'{self.fitness}_up'
        
        # 경로 설정
        self.pose_samples_folder = os.path.join(sample_csv_path, fitness)
        self.master_angle = os.path.join(master_angle_dir, f'one_tempo_{fitness}.csv')
        self.result_dir = save_result_dir
        
        
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
        
        # 각도 변화 정보 저장할 dictionary
        self.angle_info = {}
        for side in ['left', 'right']:
            self.angle_info[side] = {}
            for land in self.main_land_name[side][1:-1]:
                self.angle_info[side][land] = []
        

    def get_angle_by_temp(self, detailed_record):
        
        landmarks_dict = detailed_record['value']
        now_count = 0
        
        one_tempo_angle = copy.deepcopy(self.angle_info)
        one_tempo_angle_tmp = copy.deepcopy(self.angle_info)

        # 프레임 별 반복
        for i in range(detailed_record['total_frame']):
            
            pose_landmarks = landmarks_dict[i]

            if pose_landmarks is not None:
            # Get landmarks.
                
                np_pose_landmarks = np.array([[pose_landmarks[lmk][0] , pose_landmarks[lmk][1], pose_landmarks[lmk][2]]
                                            for lmk in self.land_name if lmk not in self.exclude_landmarks],
                                            dtype=np.float32)
                assert np_pose_landmarks.shape == ((12, 3) if self.fitness == 'squat' else (33, 3)), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = self.pose_classifier(np_pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = self.pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = self.repetition_counter(pose_classification)
            else:
                print("Error : No pose")
                sys.exit()
                
            # 각도 변화를 계산하고 one_tempo 각도 리스트에 추가하여 업데이트
            one_tempo_angle = self.update_tempo(pose_landmarks=pose_landmarks, one_tempo_angle=one_tempo_angle)     

            # Up 과 Down의 시작점을 구분하기 위해
            start_pose = f"{self.fitness}_up" if self.class_name == f"{self.fitness}_down" else f"{self.fitness}_down"
            
            # Classification 한 결과를 통해 Count를 한다.
            if repetitions_count > now_count:
                if start_pose in pose_classification_filtered and pose_classification_filtered[start_pose] == 9.999999999999998:
                
                    now_count += 1
                    for side in ['left', 'right']:
                        for i in range(1, len(self.main_land_name[side]) - 1):
                            # 한 Tempo에 저장된 각도 변화 정보를 Score 클래스의 angle_info에 저장 -> {'left_shoulder' : [[첫번째 횟수 각도 변화], [두번째 횟수 각도 변화], ...], ...}
                            self.angle_info[side][self.main_land_name[side][i]].append(one_tempo_angle[side][self.main_land_name[side][i]])
                            
                    del(one_tempo_angle)
                    one_tempo_angle = copy.deepcopy(one_tempo_angle_tmp)

        # Release MediaPipe resources.
        self.pose_tracker.close()

    def update_tempo(self, pose_landmarks, one_tempo_angle):
                     
            for side in ['left', 'right']:
                for i in range(1, len(self.main_land_name[side]) - 1):

                    a = pose_landmarks[self.main_land_name[side][i-1]]
                    b = pose_landmarks[self.main_land_name[side][i]]
                    c = pose_landmarks[self.main_land_name[side][i+1]]
                
                    angle = self.calculate_angle(a, b, c)
                    one_tempo_angle[side][self.main_land_name[side][i]].append(angle)

            return one_tempo_angle
  
    # 3차원 xyz좌표 각도 계산
    def calculate_angle(self, a, b, c):
        a = np.array(a) 
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle
            
        return angle
    
    # 전달받은 landmark 정보 json 파일로부터 dictionary 형태로 변환
    def json_to_dict(self, json_pth):
        with open(json_pth) as f:
            data = json.load(f)
        return data

    # 코사인 유사도 계산
    def similarity(self, a, b):
        len_a, len_b = len(a), len(b)
        min_len = min(len_a, len_b)
        
        indices_a = np.linspace(0, len_a - 1, min_len, dtype=int)
        indices_b = np.linspace(0, len_b - 1, min_len, dtype=int)
        
        downsampled_a = [a[i] for i in indices_a]
        downsampled_b = [b[i] for i in indices_b]

        cosine_similarity = 1 - cosine(downsampled_a, downsampled_b)

        cosine_similarity = round(cosine_similarity, 2) * 100
        
        return cosine_similarity

    # 코사인 유사도를 이용해 각도 변화에 대한 점수 계산하여 dictionaly 형태로 반환
    def calculate_score(self):

        target = pd.read_csv(self.master_angle)
        all_score = {}
        
        for side in ['left', 'right']:
            for part in self.angle_info[side].keys():
                
                
                score = [self.similarity(self.angle_info[side][part][i], target[part]) for i in range(len(self.angle_info[side][part]))]

                all_score[part] = {'set' : score, 'score' : np.mean(score)}
        
        total_score = np.mean([all_score[part]['score'] for part in all_score.keys()])
        all_score['total'] = total_score
        
        return all_score
    
    # {이름}/{날짜}/{운동이름} 디렉토리에 각도 변화 그래프 저장
    def draw_save_angle_plot(self):
        
        for side in ['left', 'right']:
            for key in self.angle_info[side].keys():
                y_line = list(itertools.chain.from_iterable(self.angle_info[side][key]))
                x_line = np.arange(len(y_line))
                
                plt.plot(x_line, y_line, label = f'{key}')
                plt.xlabel('frame')
                plt.ylabel('Angle')
                plt.legend(loc='lower left', fontsize=8)
                plt.title(f'{self.fitness} {side}')
                
            plt.savefig(os.path.join(self.result_dir, f'{self.fitness}_{side}_plot.png')) 
            plt.clf() 
        print("Complete to save angle plot")
        
    
    def save_result(self):
        # 각도 변화 그래프 저장
        self.draw_save_angle_plot()
        
        # 각도 변화에 대한 점수 계산
        all_score = self.calculate_score()
        
        # 각도 변화에 대한 점수 json 파일로 저장
        with open(os.path.join(self.result_dir, f'{self.fitness}_score.json'), 'w') as f:
            json.dump(all_score, f, indent=4)
        
        print("Complete to save score and plot")
        
        
