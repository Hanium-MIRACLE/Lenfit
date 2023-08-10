
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
        
        if fitness == 'squat':
            self.main_land_name = {'left' : ['left_shoulder', 'left_hip', 'left_knee', 'left_ankle'], 'right' : ['right_shoulder', 'right_hip', 'right_knee', 'right_ankle']}
            self.main_land_num = {'left' : [11, 23, 25, 27], 'right' : [12, 24, 26, 28]}
            
        if fitness == 'pushups':
            self.main_land_name = {'left' : ['left_wrist', 'left_elbow', 'left_shoulder', 'left_hip', 'left_knee'], 'right' : ['right_wrist', 'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']}
            self.main_land_num = {'left' : [15, 13, 11, 23, 25], 'right' : [16, 14, 12, 24, 26]}
          
        self.angle_info = {}
        
        for side in ['left', 'right']:
            self.angle_info[side] = {}
            for land in self.main_land_name[side][1:-1]:
                self.angle_info[side][land] = []
        
    
    def __call__(self, pose_landmarks, one_tempo_angle):#json_pth):
        
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
    
    def json_to_dict(self, json_pth):
        with open(json_pth) as f:
            data = json.load(f)
        return data
    
    # 2개의 3차원 좌표의 각도 계산
    def calculate_angle(self, a, b, c):
        a = np.array(a) # 2개의 3차원 좌표를 array로 변환
        b = np.array(b)
        c = np.array(c)
        
        # 각도 계산
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        # 180도를 넘으면 360도에서 빼준다.
        if angle >180.0:
            angle = 360-angle
            
        return angle

    def similarity(self, a, b):
        len_a, len_b = len(a), len(b)
        min_len = min(len_a, len_b)
        
        indices_a = np.linspace(0, len_a - 1, min_len, dtype=int)
        indices_b = np.linspace(0, len_b - 1, min_len, dtype=int)
        
        downsampled_a = [a[i] for i in indices_a]
        downsampled_b = [b[i] for i in indices_b]

        # 코사인 유사도 계산
        cosine_similarity = 1 - cosine(downsampled_a, downsampled_b)

        # 소수 2번째 자리까지 반올림 후 %로 변환
        cosine_similarity = round(cosine_similarity, 2) * 100
        
        # print("Cosine similarity: ", cosine_similarity)
        
        return cosine_similarity


    def calculate_score(self, angle_info):

        target = pd.read_csv(self.target_csv_pth)
        all_score = {}
        
        for side in ['left', 'right']:
            for part in angle_info[side].keys():
                
                score = [self.similarity(angle_info[side][part][i], target[part]) for i in range(len(angle_info[side][part]))]

                all_score[part] = np.mean(score)
                
        return all_score
    
        # Plot the information in angle info by left and right and save as two image files
    def draw_save_angle_plot(self):
        #print(sum([len(i) for i in self.angle_info['left']['left_hip']]))
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