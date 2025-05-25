import torch
import torch.nn as nn
import math

def rule_fall_detection(flatten_pose, xmin, ymin, xmax, ymax):
    # https://github.com/Alimustoofaa/YoloV8-Pose-Keypoint-Classification
    # flatten_pose format is x1,y1,conf1, x2,y2,conf2, ......
    # NOSE:           int = 0
    # LEFT_EYE:       int = 1
    # RIGHT_EYE:      int = 2
    # LEFT_EAR:       int = 3
    # RIGHT_EAR:      int = 4
    # LEFT_SHOULDER:  int = 5
    # RIGHT_SHOULDER: int = 6
    # LEFT_ELBOW:     int = 7
    # RIGHT_ELBOW:    int = 8
    # LEFT_WRIST:     int = 9
    # RIGHT_WRIST:    int = 10
    # LEFT_HIP:       int = 11
    # RIGHT_HIP:      int = 12
    # LEFT_KNEE:      int = 13
    # RIGHT_KNEE:     int = 14
    # LEFT_ANKLE:     int = 15
    # RIGHT_ANKLE:    int = 16
    left_shoulder_x = flatten_pose[15]
    left_shoulder_y = flatten_pose[16]
    right_shoulder_y = flatten_pose[19]

    left_body_x = flatten_pose[33]
    left_body_y = flatten_pose[34]
    right_body_y = flatten_pose[37]

    left_foot_y = flatten_pose[46]
    right_foot_y = flatten_pose[49]

    # len_factor is the Euclidean distance between the left shoulder and left body keypoints.
    # This gives a rough estimate of the body length (i.e., the distance between the shoulder and the torso),
    # which will be used to help determine whether the body is upright or has fallen.
    len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))

    dx = int(xmax) - int(xmin)
    dy = int(ymax) - int(ymin)
    difference = dy - dx

    # Fall Detection Logic:
    # The function checks three main conditions to detect if a fall has occurred:
    # Shoulder and foot position: If the shoulder is higher than the foot position minus len_factor,
    # and if the body's position is above the foot (considering the length), it suggests an upright position.
    # Right side condition: Similar to the left side, if the right shoulder and right body are in proper position
    # relative to the foot, it suggests the person is standing.
    # Bounding box difference: If the difference between the bounding box height (dy) and width (dx) is negative, indicating a fall.
    # If any of these conditions are met, the function returns True, indicating a fall,
    # and returns the bounding box (xmin, ymin, xmax, ymax) to highlight the fallen person.
    if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (len_factor / 2) \
            and left_shoulder_y > left_body_y - (len_factor / 2) or (
            right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (len_factor / 2)
            and right_shoulder_y > right_body_y - (len_factor / 2)) \
            or difference < 0:
        return True

    return False

# each frame could have multiple people = poses
def Rule_Model(frame):
    height, width = frame.orig_img.shape[:2]
    for pose in frame:
        bbox = pose.boxes.xyxy.squeeze(0).tolist()
        xmin, ymin, xmax, ymax = bbox

        flatten_pose = pose.keypoints.data.squeeze(0).flatten().tolist()
        for i in range(0, 51, 3):
            flatten_pose[i] /= width  # normalize x
            flatten_pose[i + 1] /= height  # normalize y
        prediction = rule_fall_detection(flatten_pose, xmin, ymin, xmax, ymax)

        if prediction:
            return prediction, bbox
    return False, None
