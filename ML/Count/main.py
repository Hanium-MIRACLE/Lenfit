from src.object import AnalysisTempoCount

# main
if __name__ == '__main__':

    fitness = 'pushups' # squat or pushups

    video_path=f'data/test_img/{fitness}.mp4'       # input video path
    out_video_dir = f'Result/{fitness}'             # output video dir
    sample_csv_path = 'data/fitness_poses_csvs_out' # Landmark Info of fitness poses
    mode = "Video"                                  # Video or Webcam

    lenfit = AnalysisTempoCount(fitness=fitness, 
                            video_path=video_path, 
                            out_video_dir = out_video_dir, 
                            sample_csv_path = sample_csv_path, 
                            mode = mode,
                            show = True)