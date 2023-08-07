from src.object import *

# main
if __name__ == '__main__':

    fitness = 'pushups' # squat or pushups

    # video_path=f'data/test_img/{fitness}.mp4'       # input video path
    # save_video_dir = f'Result/{fitness}'             # save video dir
    # sample_csv_path = 'data/fitness_poses_csvs_out' # Landmark Info of fitness poses
    # mode = "Video"                                  # Video or Webcam

    # lenfit = AnalysisTempoCount(fitness=fitness, 
    #                         video_path=video_path, 
    #                         out_video_dir = save_video_dir, 
    #                         sample_csv_path = sample_csv_path,  
    #                         mode = mode, 
    #                         show = True)
    
    name = 'jack'
    input_video_dir = f'data/test_img/{name}'       # input video path
    
    minsoo = Person(name = name, 
                    preferred_fitness = ['squat', 'pushups'], 
                    input_video_dir = input_video_dir)
    minsoo.analyze_fitness(fitness='pushups', mode='Video')