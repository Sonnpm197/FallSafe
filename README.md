Refer this repo for UI: https://github.com/Sonnpm197/FallSafe_UI

A fall detection project using ensemble of 3 models: ANN, LSTM and Rule-based

FallSafe_Keypoint_Extraction... Files to extract 17 keypoints from a body and output into csv files

17 Keypoints from yolo: https://github.com/Alimustoofaa/YoloV8-Pose-Keypoint-Classification

Video dataset used:

- MC: https://www.iro.umontreal.ca/~labimage/Dataset/
- UR: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
- Le2i: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia?resource=download

Data processing:
- In each video, for each frame we use yolopose11 to find the highest confident person in a frame, then extract keypoints for that person only, and receive
x1, y1, confidence1, x2, y2, confidence2, ...., x17, y17, confidence17, video_framenumber
- Then these points above will be appended with label as the first column
label, x1, y1, ......
- for MC dataset, FallSafe_Keypoint_Extraction_MC.ipynb is extracted to csv file
- for UR dataset, FallSafe_Keypoint_Extraction_UR.ipynb is extracted to csv file
- for Le2i dataset, FallSafe_Keypoint_Extraction_Le2i.ipynb is extracted to csv file

Models:
- can be found in nn_model.py, rule_nodel.py and sequence_model.py
- During the testing, remove the confidence from above dataset seems to perform better, so I added _no_conf models for 
using data without confidence

Training:
- can be found in FallSafe_Sequence_Training.ipynb and FallSafe_Frame_By_Frame_Training.ipynb files
- sequence for LSTM, frame_by_frame for ANN models
- Since we have all the features from yolo so only need connected layers from ANN, no need conv layers

Testing:
- Webcam test + Ensemble: FallSafe_Webcam_Test.py
- Ensemble in jupiter notebook: FallSafe_Ensemble_Perf_Test.ipynb

Serve as backend:
(main_yield_from_backend is for sending images from backend to frontend, but I didn't use this anymore. To use this just run the main method inside)


for FastAPI: 
main.py for processing video uploaded from frontend and Websocket for video streaming to detect fall in live video

uvicorn main:app --reload --host 0.0.0.0 --port 8000