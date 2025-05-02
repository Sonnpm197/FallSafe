import cv2

# Input and output file paths
input_path = './fall_videos/test/video_1.mp4'
output_path = './output/test_showing.mp4'

# Open the video file
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 output

# Create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video



    # Bottom-left position for the text
    position = (10, frame.shape[0] - 10)
    text = f'Frame: {frame_num}'

    # Draw red text
    cv2.putText(frame, text, position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),  # Red in BGR
                thickness=2)

    # Write the frame to the output video
    out.write(frame)

    # Optionally show the frame
    cv2.imshow('Processing', frame)

    frame_num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output saved to {output_path}")
