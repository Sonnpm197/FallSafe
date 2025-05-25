from collections import defaultdict, deque

import cv2

from yolo_processing import process_image_with_yolo


cap = cv2.VideoCapture(0)
sequence_length = 10

person_sequences = defaultdict(lambda: deque(maxlen=sequence_length))  # Store sequences for each person
person_cnn_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store ANN votes
person_rule_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store Rule model votes

while True:
    success, img = cap.read()

    process_image_with_yolo(img, person_sequences, person_cnn_votes, person_rule_votes)
    # print("len ", len(person_sequences))
    # print("len 1", len(person_sequences[0]))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
