import os, cv2, fnmatch, tqdm

from utils.video_utils import frame_gen_from_video
from utils.general_utils import try_wrapper


input_folder = "../../data/data/"
output_folder = "../../data/resized_data/"
os.makedirs(output_folder, exist_ok=True)
log_path = os.path.join(output_folder, "error_log.txt")

OUTPUT_SIZE=768
OUTPUT_FPS=24

def process_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("basename", basename)
    # print("width", width)
    # print("height", height)
    # print("frames_per_second", frames_per_second)
    # print("num_frames", num_frames)

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(OUTPUT_FPS),
        frameSize=(OUTPUT_SIZE, OUTPUT_SIZE),
        isColor=True,
    )

    frame_gen = frame_gen_from_video(video)
    
    time_per_frame_original = 1 / frames_per_second
    time_per_frame_target = 1 / OUTPUT_FPS
    accumulated_time = 0

    for frame in frame_gen:
        accumulated_time += time_per_frame_original

        if accumulated_time >= time_per_frame_target:
            accumulated_time -= time_per_frame_target

            if width < height:
                square_image = cv2.copyMakeBorder(
                    frame, 
                    top=0, 
                    bottom=0, 
                    left=(height-width)//2, 
                    right=(height-width+1)//2, 
                    borderType=cv2.BORDER_REFLECT_101
                )
            else:
                square_image = frame[:, width//2-height//2: width//2+(height+1)//2]
            
            resized_image = cv2.resize(square_image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation = cv2.INTER_LINEAR)

            output_file.write(resized_image)

    video.release()
    output_file.release()

    # video2 = cv2.VideoCapture(os.path.join(output_folder, basename))

    # width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames_per_second = video2.get(cv2.CAP_PROP_FPS)
    # num_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    # print("basename", basename)
    # print("width", width)
    # print("height", height)
    # print("frames_per_second", frames_per_second)
    # print("num_frames", num_frames)

    # video2.release()

input_files = sorted(os.listdir(input_folder))
output_files = os.listdir(output_folder)

for filename in tqdm.tqdm(input_files):
    if fnmatch.fnmatch(filename, '*-original.mp4'):
        if filename in output_files:
            continue

        input_path = os.path.join(input_folder, filename)
        try_wrapper(lambda: process_video(input_path), filename, log_path)

# python dataset_preprocessing/video_sampling_resizing.py