
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
import copy
import os

from scipy.spatial.distance import cosine

class Score:
    def __init__(self, target_dir, out_plot_dir, fitness):
        
        self.fitness = fitness
        self.target_csv_pth = os.path.join(target_dir, f'one_tempo_{fitness}.csv')
        self.out_plot_dir = out_plot_dir
        
        # 각 운동별 주요 관절 정보
        if fitness == 'squat':
            self.main_land_name = {'left' : ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle'], 'right' : ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle']}
            self.main_land_num = {'left' : [11, 23, 25, 27], 'right' : [12, 24, 26, 28]}
            
        if fitness == 'pushups':
            self.main_land_name = {'left' : ['left_wrist', 'left_elbow', 'left_shoulder', 'left_hip', 'left_knee'], 'right' : ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']}
            self.main_land_num = {'left' : [15, 13, 11, 23, 25], 'right' : [16, 14, 12, 24, 26]}
          
        # 각도 변화 정보 저장할 dictionary
        self.angle_info = {}
        
        # 위 dictionary 초기화
        for side in ['left', 'right']:
            self.angle_info[side] = {}
            for land in self.main_land_name[side][1:-1]:
                self.angle_info[side][land] = []
        
    
    def __call__(self, pose_landmarks, one_tempo_angle):#landmark_json_pth):
        
        #pose_landmarks = self.json_to_dict(json_pth)
        
        for side in ['left', 'right']:
            for i in range(1, len(self.main_land_name[side]) - 1):
                if self.fitness == 'squat':
                    a = pose_landmarks[(i - 1) * 2 + (1 if side == 'right' else 0)]
                    b = pose_landmarks[i * 2 + (1 if side == 'right' else 0)]
                    c = pose_landmarks[(i + 1) * 2 + (1 if side == 'right' else 0)]
                    
                else:
                    a = pose_landmarks[self.main_land_num[side][i-1]]
                    b = pose_landmarks[self.main_land_num[side][i]]
                    c = pose_landmarks[self.main_land_num[side][i+1]]
            
                angle = self.calculate_angle(a, b, c)
                one_tempo_angle[side][self.main_land_name[side][i]].append(angle)

        return one_tempo_angle
    
    # 전달받은 landmark 정보 json 파일로부터 dictionary 형태로 변환
    def json_to_dict(self, json_pth):
        with open(json_pth) as f:
            data = json.load(f)
        return data
    
    # 각도 계산
    def calculate_angle(self, a, b, c):
        a = np.array(a) 
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle >180.0:
            angle = 360-angle
            
        return angle

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
    def calculate_score(self, angle_info):

        target = pd.read_csv(self.target_csv_pth)
        all_score = {}
        
        for side in ['left', 'right']:
            for part in angle_info[side].keys():
                
                score = [self.similarity(angle_info[side][part][i], target[part]) for i in range(len(angle_info[side][part]))]

                all_score[part] = np.mean(score)
        
        total_score = np.mean(list(all_score.values()))
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
                
            plt.savefig(os.path.join(self.out_plot_dir, f'{self.fitness}_{side}.png')) 
            plt.clf() 
        print("Complete to save angle plot")