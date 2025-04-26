import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# plan:

# 1. test the camera if it is working
# 2. if it is working dectect hand, if not go to 1
# 3. crop the image + center
# 4. make img same size: if h > w => strecht w, if w > h => strecht to fit the imgsize
# 5. capture the img and save the img 

# 1
cap = cv2.VideoCapture(0) # 0 is the id number for web cam
detector = HandDetector(maxHands=1) # 2

offset = 20
imgSize = 300

# save image when press button
folder =  "Data/Yes" # 5
counter = 0 # count the number of saved img
while True:
    success, img = cap.read() # 1
    hands, img = detector.findHands(img) # 2

    if hands: # 3
        hand = hands[0] # hand[0] = 1 hand
        x,y,w,h = hand['bbox'] # give the value of where the hand is

        # 3 make sure that it is center
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8) * 255

       #crop the image
       # 3 
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imageCropShape = imgCrop.shape # contain 3 values: height, width, channel
        
        # calculate to make the image fit inside the white space

        # 4
        # if the height is greater than width, stretch the height to 300 and calculate the width value
        # if the width is greater than height, stretch the width to 300 and calculate the height value
        aspectRatio = h / w
        # for height
        if aspectRatio > 1:
            k = imgSize / h # stretch the height
            wCal = math.ceil(k * w) # calculate width

            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imageResizeShape = imgResize.shape

            # make the img center in the white
            wGap = math.ceil((imgSize - wCal) / 2)
            # put image white into image crop
            imgWhite[0:imageResizeShape[0], wGap:wCal + wGap] = imgResize
        # for width
        else:
            k = imgSize / w # stretch the width
            hCal = math.ceil(k * h) # calculate width

            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imageResizeShape = imgResize.shape

            # make the img center in the white
            hGap = math.ceil((imgSize - hCal) / 2)
            # put image white into image crop
            imgWhite[hGap:hCal + hGap, :] = imgResize


        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", img) # 1
    key = cv2.waitKey(2) # 1

    # if press the s key it will save the img
    if key == ord("s"): # 5
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print(counter)