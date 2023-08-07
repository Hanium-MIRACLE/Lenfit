from src.srcs import BootstrapHelper

bootstrap = BootstrapHelper(images_in_folder="data/train_img", 
                           images_out_folder="data/train_img_out",
                           csvs_out_folder="data/squat",
                           fitness="squat")

bootstrap.bootstrap()