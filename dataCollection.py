from time import time
import os
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# ========================= SETTINGS ========================= #
classID = 0  # 0 = fake, 1 = real
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35  # Higher value = clearer image

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6

# Ensure the output folder exists
os.makedirs(outputFolderPath, exist_ok=True)

# ========================= INITIALIZATION ========================= #
cap = cv2.VideoCapture(0)  # Change to (1) if using an external camera
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # Stores blur status of detected faces
    listInfo = []  # Stores label data for detected faces

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # ------ Check confidence score --------
            if score > confidence:
                # ------ Add offset to face detection --------
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # Ensure values are within bounds
                x, y = max(0, x), max(0, y)
                w, h = max(0, w), max(0, h)

                # ------ Ensure cropping is valid --------
                if y + h <= img.shape[0] and x + w <= img.shape[1]:
                    imgFace = img[y:y + h, x:x + w]
                else:
                    continue  # Skip invalid cropping

                cv2.imshow("Face", imgFace)

                # ------ Compute Blurriness --------
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                listBlur.append(blurValue > blurThreshold)

                # ------ Normalize Values --------
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Ensure normalized values are within 0-1
                xcn, ycn = min(1, xcn), min(1, ycn)
                wn, hn = min(1, wn), min(1, hn)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ------ Draw rectangle and put text --------
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), (255, 0, 0), 3)  # FIXED
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                   scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                       scale=2, thickness=3)

        # ------ Save Images & Labels --------
        if save and all(listBlur):  # Only save if all faces are clear
            timeNow = str(time()).replace('.', '')  # Unique timestamp

            # Save Image
            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

            # Save Label Text File
            with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                f.writelines(listInfo)

    cv2.imshow("Image", imgOut)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
