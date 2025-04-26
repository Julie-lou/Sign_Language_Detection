
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# === Fix DepthwiseConv2D error by overriding the class ===
class FixedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove unsupported argument
        super().__init__(*args, **kwargs)

custom_objects = {'DepthwiseConv2D': FixedDepthwiseConv2D}

# === Load labels from labels.txt ===
def load_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(' ', 1)[1] for line in lines]

labels = load_labels("Model/labels.txt")

# === Load model with custom DepthwiseConv2D fix ===
model = load_model("Model/keras_model.h5", custom_objects=custom_objects, compile=False)

# === Webcam and hand detection setup ===
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# === Image processing constants ===
offset = 20
imgSize = 240
modelInputSize = 224  # Input shape your model expects

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Resize to match model input shape and normalize
        imgInput = cv2.resize(imgWhite, (modelInputSize, modelInputSize))
        imgInput = imgInput.astype(np.float32) / 255.0
        imgInput = np.expand_dims(imgInput, axis=0)

        # Predict using model
        prediction = model.predict(imgInput, verbose=0)
        index = np.argmax(prediction)
        confidence = prediction[0][index]

        label = f"{labels[index]} ({confidence*100:.1f}%)"
        print("Prediction:", label)

        # Display label on screen
        cv2.rectangle(imgOutput, (x - offset, y - offset - 30), (x + w + offset, y - offset), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow("Cropped", imgCrop)
        cv2.imshow("White Background", imgWhite)

    cv2.imshow("Sign Language Detection", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




