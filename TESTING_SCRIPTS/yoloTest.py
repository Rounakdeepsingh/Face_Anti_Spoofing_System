from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load YOLO model (make sure the path to your model is correct)
model = YOLO("../models/yolov8n.pt")

# Class names for detection (should match the dataset used to train the model, e.g., COCO dataset)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
              "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
              "toothbrush"]

# Frame timing for FPS calculation
prev_frame_time = 0

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break  # Exit if the frame is not successfully read

    # Perform object detection on the frame
    results = model(img, stream=True)

    # Iterate through detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Get confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Get class index
            cls = int(box.cls[0])

            # Display class name and confidence on the frame
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

    # Calculate FPS (frames per second)
    new_frame_time = time.time()
    if prev_frame_time != 0:  # Avoid division by zero
        fps = 1 / (new_frame_time - prev_frame_time)
        print(f"FPS: {fps:.2f}")

    prev_frame_time = new_frame_time

    # Display the resulting frame
    cv2.imshow("Image", img)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Use 1 for real-time frame display
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
