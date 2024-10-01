import os, cv2

from utils.video_utils import frame_from_video

input_path = "../../data/data/0012b1de-a058-43f3-9788-f662afc43070-original.mp4"
output_folder = "../../data/resized_data/"
os.makedirs(output_folder, exist_ok=True)

OUTPUT_SIZE=768

def process_video(input_path):
    video = cv2.VideoCapture(input_path)

    basename = os.path.basename(input_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("basename", basename)
    print("width", width)
    print("height", height)
    print("frames_per_second", frames_per_second)
    print("num_frames", num_frames)

    output_file = cv2.VideoWriter(
        filename=os.path.join(output_folder, basename),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=float(frames_per_second),
        frameSize=(OUTPUT_SIZE, OUTPUT_SIZE),
        isColor=True,
    )

    frame_gen = frame_from_video(video)
    
    for frame in frame_gen:
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

process_video(input_path)

# python dataset_preprocessing/video_sampling_resizing.py