import sys
sys.path.append(".")

from utils.video_utils import hash_file

# print("file1", hash_file("../../data/resized_data/6c6aa0b1-b3b3-4768-b87b-a020743213fc-original.mp4"))
# print("file2", hash_file("../../data/resized_data/75ec5951-d644-48aa-87c2-87cfb61aa7c6-original.mp4"))
# print("same", hash_file("../../data/resized_data/6c6aa0b1-b3b3-4768-b87b-a020743213fc-original.mp4") == hash_file("../../data/resized_data/75ec5951-d644-48aa-87c2-87cfb61aa7c6-original.mp4"))

print("file1", hash_file("../../data/resized_data/df5afa6a-b7a2-485e-ae12-e3d045e4ebc0-original.mp4"))
print("file2", hash_file("../../data/resized_data/00368efb-8457-4425-9789-3a1ae302b1ae-original.mp4"))
print("same", hash_file("../../data/resized_data/df5afa6a-b7a2-485e-ae12-e3d045e4ebc0-original.mp4") == hash_file("../../data/resized_data/00368efb-8457-4425-9789-3a1ae302b1ae-original.mp4"))

# python dataset_preprocessing/remove_duplicate_videos.py