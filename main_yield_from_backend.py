import mimetypes
import os
import threading
from collections import defaultdict, deque

import cv2
from flask import Flask, request, send_file, send_from_directory
from flask import Response
from flask_cors import CORS
from flask_socketio import SocketIO

from yolo_processing import process_video_ensemble_model, process_image_with_yolo

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", threaded=True)  # Allow all origins
CORS(app)

@app.route('/videos/<path:filename>')
def serve_video(filename):
    return send_from_directory('uploads', filename)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    filename = file.filename
    ext = os.path.splitext(filename)[1]  # e.g., ".avi"

    input_path = f'./uploads/input{ext}'
    output_path = f'./uploads/output{ext}'

    file.save(input_path)
    process_video_ensemble_model(input_path, output_path)  # Assume output format matches input

    return send_file(
        output_path,
        as_attachment=False,  # Optional — triggers download
        mimetype=mimetypes.guess_type(output_path)[0]
    )

is_streaming = False

person_sequences = defaultdict(lambda: deque(maxlen=10))  # Store sequences for each person
person_cnn_votes = defaultdict(lambda: deque(maxlen=10))  # Store ANN votes
person_rule_votes = defaultdict(lambda: deque(maxlen=10))  # Store Rule model votes

def generate_video_feed():
    # process_image_with_yolo(frame, person_sequences, person_cnn_votes, person_rule_votes)
    cap = cv2.VideoCapture(0)
    while True:
        if not is_streaming:
            break
        ret, frame = cap.read()
        if not ret:
            break

        global person_sequences, person_cnn_votes, person_rule_votes
        process_image_with_yolo(frame, person_sequences, person_cnn_votes, person_rule_votes)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    print("Stopped here")
    global is_streaming
    is_streaming = False

    return "Stopped", 200

@app.route('/video_feed')
def video_feed():
    global is_streaming
    if not is_streaming:
        is_streaming = True

    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
