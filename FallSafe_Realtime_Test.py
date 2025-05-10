from ultralytics import YOLO
import torch
import os
import cv2
import torch
from collections import defaultdict, deque
from utils import is_none_or_empty
from rule_model import rule_fall_detection
from sequence_model import LSTM_Model_2
from nn_model import NN_Model_NO_CONF_2
from rule_model import rule_fall_detection

model = YOLO("yolo11n-pose.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(0)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
sequence_length = 10

person_sequences = defaultdict(lambda: deque(maxlen=sequence_length))  # Store sequences for each person
person_cnn_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store ANN votes
person_rule_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store Rule model votes

lstm_model_no_conf2_path = "./model/lstm_model_no_conf2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lstm_model_no_conf2 = LSTM_Model_2(input_dim=34)
lstm_model_no_conf2.load_state_dict(torch.load(lstm_model_no_conf2_path, map_location=device))
lstm_model_no_conf2.to(device)
lstm_model_no_conf2.eval()

# Define nn model
nn_model_no_conf_2 = NN_Model_NO_CONF_2()
nn_model_no_conf_2.load_state_dict(torch.load('./model/nn_model_no_conf2.pth'))

nn_model_no_conf_2.eval()
nn_model_no_conf_2.to(device)

using_conf = False

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for frame_number, result in enumerate(results):
        frame = result.orig_img.copy()
        boxes = result.boxes
        keypoints = result.keypoints

        if is_none_or_empty(boxes) or is_none_or_empty(keypoints):
            continue

        for index in range(len(boxes)):
            conf_value = float(boxes.conf[index])
            if conf_value < 0.5:
                continue

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
                cnn_pred = nn_model_no_conf_2(input_tensor).item()

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
                    lstm_output = lstm_model_no_conf2(sequence_tensor)  # Output shape: (1, 2) for binary classification
                    lstm_pred = torch.argmax(lstm_output, dim=1).item()

                # Perform majority voting
                cnn_seq = person_cnn_votes[person_id]
                rule_seq = person_rule_votes[person_id]

                cnn_label = int(sum(cnn_seq) >= (sequence_length // 2))
                rule_label = int(sum(rule_seq) >= (sequence_length // 2))
                lstm_label = int(lstm_pred > 0.5)

                # Voting mechanism: majority rule for combining CNN, Rule, and LSTM predictions
                votes = [cnn_label, rule_label, lstm_label]
                final_prediction = int(sum(votes) >= 2)

                prediction_label = "FALL" if final_prediction == 1 else "SAFE"
                color = (0, 0, 255) if prediction_label == "FALL" else (0, 255, 0)  # Red for fall, green for safe

                # Draw the bounding box
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

                # Draw the label text above the box
                label_position = (int(xmin), int(ymin) - 10)
                cv2.putText(frame, prediction_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow('Webcam', frame)
    # cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
