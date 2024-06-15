import numpy as np
from utils.video import video_to_images
from pathlib import Path
import cv2
from tqdm import tqdm

EXPI_SUBJECTS = ['acro1', 'acro2']
CHI3D_SUBJECTS = ['s02', 's03', 's04']
HI4D_SUBJECTS = ['.']

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

def main():
    subjects = None
    data_names = ["chi3d", "expi", "hi4d"]
    for data_name in data_names:
        print(f"Processing {data_name} data")
        if data_name == "chi3d":
            subjects = CHI3D_SUBJECTS
            data_name = "chi3d/train"
        elif data_name == "expi":
            subjects = EXPI_SUBJECTS
        elif data_name == "hi4d":
            subjects = HI4D_SUBJECTS

        for subj_name in subjects:
            vids_path = f'./data/videos/{data_name}/{subj_name}/videos'
            videos = sorted(Path(vids_path).glob('*.mp4'))
            for vid_file in tqdm(videos):
                vname = vid_file.stem.replace(" ", "_")
                output_folder = f'./data/videos/{data_name}/{subj_name}/images/{vname}'
                if Path(output_folder).exists():
                    continue
                vid_file_str = str(vid_file)
                fps = int(get_video_fps(vid_file_str))
                video_to_images(vid_file_str, img_folder=output_folder, return_info=False, fps=fps, ext='jpg')


if __name__ == "__main__":
    main()
