from ultralytics import YOLO
import pandas as pd
import os
import cv2

# I run this because run jupiter on this cause crash

# GENERATE keypoints here
model = YOLO("yolo11n-pose.pt")  # load an official model

input_folder = './fall_videos/ur_videos_processed'

all_keypoints = []

def is_none_or_empty(x):
    return x is None or len(x) == 0

# normalize data according to the size of videos in different dataset
def normalize(list, width, height, offset):
    for i in range(offset, 51 + offset, 3):  # 17 keypoints, each keypoint is x, y and confidence
        list[i] = float(list[i]) / width
        list[i + 1] = float(list[i + 1]) / height

for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):
        full_path = os.path.join(input_folder, filename)
        print("Processing {}".format(full_path))
        results = model.track(source=full_path, stream=True)

        for frame_number, result in enumerate(results): # each is a frame
            height, width = result.orig_img.shape[:2]
            boxes = result.boxes  # Detections
            keypoints = result.keypoints  # If using a pose model

            if is_none_or_empty(boxes) or is_none_or_empty(keypoints):
                continue

            best_index = None
            best_conf = 0.0

            # inside a frame can have multiple people but only take the highest confidence
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                if cls == 0 and conf > best_conf:
                    best_conf = conf
                    best_index = i

            # print("best confidence: ", best_conf)

            # keypoints.data is a torch.Tensor of shape (num_detections, num_keypoints, 3)
            keypoints_tensor = keypoints.data[best_index]

            keypoints_np = keypoints_tensor.cpu().detach().numpy()

            base_name = os.path.splitext(filename)[0]
            tag = f"{base_name}_{frame_number}"

            flat = keypoints_np.flatten().tolist()
            normalize(flat, width, height, 0)

            # Extract bounding box coordinates
            bbox = boxes[best_index].xyxy.squeeze(0).tolist()  # [xmin, ymin, xmax, ymax]

            # Combine with flattened keypoints and tag
            flat_with_tag = bbox + flat + [tag]

            # Store the result
            all_keypoints.append(flat_with_tag)
    break

df = pd.DataFrame(all_keypoints)
df.to_csv('./csv_data/ur_keypoints_yolo1.csv', index=False, header=False)