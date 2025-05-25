from collections import defaultdict, deque

import cv2
import torch
from ultralytics import YOLO

from rule_model import rule_fall_detection
from utils import is_none_or_empty
from sequence_model import LSTM_Model_2
from nn_model import NN_Model_NO_CONF_2

# Define models
yolo_model = YOLO("yolo11n-pose.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lstm_model = LSTM_Model_2(input_dim=34)
lstm_model.load_state_dict(torch.load("./model/lstm_model_no_conf2.pth", map_location=device))
lstm_model.to(device)
lstm_model.eval()

nn_model = NN_Model_NO_CONF_2()
nn_model.load_state_dict(torch.load('./model/nn_model_no_conf2.pth'))
nn_model.to(device)
nn_model.eval()

sequence_length = 10
using_conf = False

def process_single_frame(input_extraction, output_image, person_sequences, person_cnn_votes, person_rule_votes):
    height, width = input_extraction.orig_img.shape[:2]
    boxes = input_extraction.boxes
    keypoints = input_extraction.keypoints

    person_fall = False
    if is_none_or_empty(boxes) or is_none_or_empty(keypoints):
        return person_fall

    for index in range(len(boxes)):
        conf_value = float(boxes.conf[index])
        if conf_value < 0.5:
            return person_fall

        # Track each person's keypoints and store predictions
        keypoints_tensor = keypoints.data[index]
        keypoints_np = keypoints_tensor.cpu().detach().numpy()
        flat = keypoints_np.flatten().tolist()
        flat_for_rule = keypoints_np.flatten().tolist()

        for i in range(0, 51, 3):
            flat[i] = float(flat[i]) / float(width)
            flat[i + 1] = float(flat[i + 1]) / float(height)

        if not using_conf:
            flat = [flat[i] for i in range(len(flat)) if i % 3 != 2]

        input_tensor = torch.tensor(flat, dtype=torch.float32).view(1, 1, len(flat)).to(device)
        box_id = boxes[index].id
        if box_id is not None:
            person_id = int(box_id.item())  # works for tensor([1.])
        else:
            person_id = index  # fallback to index

        # Predict using CNN model
        with torch.no_grad():
            cnn_pred = nn_model(input_tensor).item()

        # Predict using Rule Model
        xmin, ymin, xmax, ymax = boxes[index].xyxy.squeeze(0).tolist()
        rule_pred_bool = rule_fall_detection(flat_for_rule, xmin, ymin, xmax, ymax)

        # Record CNN and Rule predictions
        person_cnn_votes[person_id].append(int(cnn_pred > 0.5))
        person_rule_votes[person_id].append(int(rule_pred_bool))

        # Collect the sequence of keypoints for LSTM model (if enough frames are collected)
        person_sequences[person_id].append(flat)

        # if we have a sequence of len sequence_length, then start processing ensembling
        if len(person_sequences[person_id]) == sequence_length:
            # LSTM model prediction
            sequence_tensor = (torch.tensor(person_sequences[person_id], dtype=torch.float32)
                               .view(1, sequence_length, len(flat)).to(device))
            with torch.no_grad():
                lstm_output = lstm_model(sequence_tensor)  # Output shape: (1, 2) for binary classification
                lstm_pred = torch.argmax(lstm_output, dim=1).item()

            # Perform majority voting
            cnn_seq = person_cnn_votes[person_id]
            rule_seq = person_rule_votes[person_id]

            cnn_label = int(sum(cnn_seq) >= (sequence_length // 2))
            rule_label = int(sum(rule_seq) >= (sequence_length // 2))
            lstm_label = int(lstm_pred > 0.5)

            # Voting mechanism: majority rule for combining CNN, Rule, and LSTM predictions
            votes = [cnn_label, rule_label, lstm_label]
            # votes = [cnn_label, lstm_label]
            final_prediction = int(sum(votes) >= 2)
            # final_prediction = cnn_label

            prediction_label = "SUSPICIOUS STATUS" if final_prediction == 1 else "SAFE"
            color = (0, 165, 255) if prediction_label == "SUSPICIOUS STATUS" else (0, 255, 0)  # Red for fall, green for safe

            if prediction_label == "SUSPICIOUS STATUS":
                person_fall = True

            # Draw the bounding box
            cv2.rectangle(output_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

            # Draw the label text above the box
            label_position = (int(xmin), int(ymin) - 10)
            cv2.putText(output_image, prediction_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return person_fall

def process_image_with_yolo(input_img, person_sequences, person_cnn_votes, person_rule_votes, display=True):
    results = yolo_model(input_img, stream=True, verbose=False)
    person_fall = False
    for frame_number, frame_extraction in enumerate(results):
        person_fall = process_single_frame(frame_extraction, input_img, person_sequences, person_cnn_votes, person_rule_votes)
    if display:
        cv2.imshow('Webcam', input_img)
    return person_fall

def process_video_ensemble_model(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use the same codec as input (if available)
    input_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((input_fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Use mp4v if input codec is not compatible
    if codec_str.strip() == "":
        codec_str = 'mp4v'

    fourcc = cv2.VideoWriter_fourcc(*codec_str)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {input_path}")
    results = yolo_model.track(source=input_path, stream=True, verbose=False)

    person_sequences = defaultdict(lambda: deque(maxlen=sequence_length))  # Store sequences for each person
    person_cnn_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store ANN votes
    person_rule_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store Rule model votes

    for frame_number, frame_extraction in enumerate(results):
        frame = frame_extraction.orig_img.copy()
        process_single_frame(frame_extraction, frame, person_sequences, person_cnn_votes, person_rule_votes)
        out.write(frame)

    out.release()
    cap.release()
    print(f"Saved annotated video to: {output_path}")