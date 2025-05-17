from collections import defaultdict, deque

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
import cv2
import numpy as np
import base64

from yolo_processing import process_video_ensemble_model, process_image_with_yolo

app = FastAPI()

# CORS setup (if frontend is on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "videos"
PROCESSED_DIR = "processed"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ----------- Upload API -----------

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_DIR}/{video_id}_{file.filename}"
    output_path = f"{PROCESSED_DIR}/{video_id}_processed.mp4"

    # Save uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_video_ensemble_model(input_path, output_path)

    return {"output_video": output_path}


# ----------- Static File Serving -----------

@app.get("/processed/{filename}")
async def get_processed_video(filename: str):
    path = f"{PROCESSED_DIR}/{filename}"
    if os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"error": "File not found"})


# ----------- WebSocket Streaming -----------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_length = 10
    person_sequences = defaultdict(lambda: deque(maxlen=sequence_length))  # Store sequences for each person
    person_cnn_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store ANN votes
    person_rule_votes = defaultdict(lambda: deque(maxlen=sequence_length))  # Store Rule model votes
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes to NumPy array (JPEG)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # TODO: Your bounding box drawing
            person_fall = process_image_with_yolo(frame, person_sequences, person_cnn_votes, person_rule_votes)

            # Encode frame to JPEG and then base64
            _, jpeg = cv2.imencode('.jpg', frame)
            b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            await websocket.send_json({
                "fall": person_fall,
                "frame": b64_frame,
            })

    except WebSocketDisconnect:
        print("WebSocket disconnected")