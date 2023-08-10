from matplotlib import pyplot as plt
import csv
import os

def show_image(img, figsize=(10, 10)):
  """Shows output PIL image."""
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.show()
  

def dump_for_the_app(fitness): # dump the csv files for the app
  pose_samples_folder = f'data/fitness_poses_csvs_out/{fitness}'
  pose_samples_csv_path = f'fitness_csv/{fitness}_poses_csvs_out.csv'
  file_extension = 'csv'
  file_separator = ','

  # Each file in the folder represents one pose class.
  file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

  with open(pose_samples_csv_path, 'w') as csv_out:
    csv_out_writer = csv.writer(csv_out, delimiter=file_separator, quoting=csv.QUOTE_MINIMAL)
    for file_name in file_names:
      # Use file name as pose class name.
      class_name = file_name[:-(len(file_extension) + 1)]

      # One file line: `sample_00001,x1,y1,x2,y2,....`.
      with open(os.path.join(pose_samples_folder, file_name)) as csv_in: 
        csv_in_reader = csv.reader(csv_in, delimiter=file_separator) # read the csv file
        for row in csv_in_reader: 
          row.insert(1, class_name)
          csv_out_writer.writerow(row)
  


