from src import show_image, dump_for_the_app
from object import *
import time

from mediapipe.python.solutions import drawing_utils as mp_drawing

# Specify your video name and target pose class to count the repetitions.
class AnalysisTempoCount:
    def __init__(self, fitness, video_path, out_video_dir, sample_csv_path, mode = 'Video', show = True):
        
        assert fitness in ['pushups', 'squat'], 'Unexpected fitness: {}'.format(fitness)
        assert mode in ['Video', 'Webcam'], 'Unexpected mode: {}'.format(mode)
        
        # Path to video file.
        self.video_path = video_path
        self.fitness = fitness
        self.out_video_path = out_video_dir
        self.class_name = f'{self.fitness}_down'
        self.pose_samples_folder = os.path.join(sample_csv_path, fitness)
        
        # Video parameters.
        self.video = cv2.VideoCapture(self.video_path) if mode == 'Video' else cv2.VideoCapture(0)
        
        self.video_n_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize tracker.
        self.pose_tracker = mp_pose.Pose()
        # Initialize embedder.
        self.pose_embedder = FullBodyPoseEmbedder()
        # Initialize classifier.
        self.pose_classifier = PoseClassifier(
                                    pose_samples_folder=self.pose_samples_folder,
                                    pose_embedder=self.pose_embedder,
                                    top_n_by_max_distance=30,
                                    top_n_by_mean_distance=10)
        
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
        
        if not self.video.isOpened():
            print("Could not open video")
            sys.exit()
        if mode == 'Webcam':
            self.Count_analysis_tempo(show)
        else:
            self.analysis_tempo(show)
        
    def analysis_tempo(self, show=False):
        
        frame_idx = 0
        output_frame = None
        
        n_out_video = 0
        out_video_name = f'{self.fitness}_anlysis_{n_out_video}.mp4'
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
                # Get landmarks.
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                for lmk in pose_landmarks.landmark], dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

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
                    
                start_pose = f"{self.fitness}_up" if self.class_name == f"{self.fitness}_down" else f"{self.fitness}_down"
                    
                if repetitions_count > n_out_video:
                    if start_pose in pose_classification_filtered and pose_classification_filtered[start_pose] == 9.999999999999998:
                    
                        n_out_video += 1
                        out_video.release()
                        out_video_name = f'{self.fitness}_anlysis_{n_out_video}.mp4'
                        out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))
                
                
                # Draw classification plot and repetition counter.
                output_frame = self.pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=None,#pose_classification_filtered,
                    repetitions_count=repetitions_count)

                # Draw Classification Result on the frame.
                
                output_frame = np.array(output_frame)
                #cv2.putText(output_frame, f"{pose_classification_filtered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Save the output frame.
                
                out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                if show:
                    cv2.imshow('Output Video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

                    # Exit the loop if 'q' is pressed.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                pbar.update()


        # Save and Close Video
        out_video.release()
        self.video.release()
        
        print("Save video path :", self.out_video_path)
        
        # Release MediaPipe resources.
        self.pose_tracker.close()

    def Count_analysis_tempo(self, show=False):
        
        # Counting
        start_time = time.time()
        countdown_over = False
        
        frame_idx = 0
        output_frame = None
        
        n_out_video = 0
        out_video_name = f'{self.fitness}_anlysis_{n_out_video}.mp4'
        out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))
        
        with tqdm.tqdm(total=self.video_n_frames, position=0, leave=True) as pbar:
            while True:
                # Get next frame of the video.
                success, input_frame = self.video.read()
                if not success:
                    break
                
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                
                # Draw countdown.
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
    
                # After countdown is over.
                else:
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
                        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                                    for lmk in pose_landmarks.landmark], dtype=np.float32)
                        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

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
                        
                    start_pose = f"{self.fitness}_up" if self.class_name == f"{self.fitness}_down" else f"{self.fitness}_down"
                        
                    if repetitions_count > n_out_video:
                        if start_pose in pose_classification_filtered and pose_classification_filtered[start_pose] == 9.999999999999998:
                        
                            n_out_video += 1
                            out_video.release()
                            out_video_name = f'{self.fitness}_anlysis_{n_out_video}.mp4'
                            out_video = cv2.VideoWriter(os.path.join(self.out_video_path, out_video_name), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (self.video_width, self.video_height))
                    
                    
                    # Draw classification plot and repetition counter.
                    output_frame = self.pose_classification_visualizer(
                        frame=output_frame,
                        pose_classification=pose_classification,
                        pose_classification_filtered=None,#pose_classification_filtered,
                        repetitions_count=repetitions_count)

                    # Draw Classification Result on the frame.
                    
                    output_frame = np.array(output_frame)
                    #cv2.putText(output_frame, f"{pose_classification_filtered}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Save the output frame.
                    
                    out_video.write(cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

                if show:
                    cv2.imshow('Output Video', cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
                    # Exit the loop if 'q' is pressed.
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                pbar.update()


        # Save and Close Video
        out_video.release()
        self.video.release()
        
        print("Save video path :", self.out_video_path)
        
        # Release MediaPipe resources.
        self.pose_tracker.close()


# main
if __name__ == '__main__':

    fitness = 'pushups' # squat or pushups

    video_path=f'data/test_img/{fitness}_1.mp4'       # input video path
    out_video_dir = f'Result/{fitness}'             # output video dir
    sample_csv_path = 'data/fitness_poses_csvs_out' # Landmark Info of fitness poses
    mode = "Webcam"                                  # Video or Webcam

    lenfit = AnalysisTempoCount(fitness=fitness, 
                            video_path=video_path, 
                            out_video_dir = out_video_dir, 
                            sample_csv_path = sample_csv_path, 
                            mode = mode,
                            show = True)