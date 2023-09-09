from src.score import *
import argparse

def json_to_dict(json_pth):
        with open(json_pth) as f:
            data = json.load(f)
        return data

# main
if __name__ == '__main__':

    # 터미널에서 --video 옵션으로 json파일, csv파일 경로를 입력받음
    parser = argparse.ArgumentParser(description='Score')
    parser.add_argument('--json', type=str, default='sample_json.json', help='Landmark json file path')
    parser.add_argument('--pose', type=str, default='fitness_poses_csvs_out', help='Pose sample csv file path')
    parser.add_argument('--master', type=str, default='angle_info', help='Master\'s angle csv file path')
    parser.add_argument('--out', type=str, default='Result', help='Save result dir')
    parser.add_argument('--fitness', type=str, default='squat', help='fitness type')
    
    args = parser.parse_args()
    
    
    # 입력받은 옵션들 출력
    print('json file path: ', args.json)
    print('pose sample csv file path: ', args.pose)
    print('master\'s angle csv file path: ', args.master)
    print('save result dir: ', args.out)
    print('fitness type: ', args.fitness)
    
    score = TempoScore(fitness=args.fitness, 
                       master_angle_dir=args.master,
                       sample_csv_path=args.pose,
                        save_result_dir=args.out)
    
    score.get_angle_by_temp(json_to_dict(args.json))
    score.save_result()
    
     
    