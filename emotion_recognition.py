import cv2 as cv
import numpy as np
import argparse

def auto_canny(image, sigma=0.33):
    image = cv.GaussianBlur(image, (3, 3), 0)
	# compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    eye_threshold_y = 1
    mouth_threshold_y = 0.9
    eyebrows_threshold_y = 0.5
    face_threshold_x = 0.5

    infoROI = {}

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        end_w, end_h = (x + w, y + h)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:end_h,x:end_w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        mouth = mouth_cascade.detectMultiScale(faceROI)
        cv.imshow("face", faceROI)

        for (x2,y2,w2,h2) in eyes:
            end_x, end_y = (x + x2 + w2, y + y2 + h2)
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            if eye_center[1] < y + eye_threshold_y * h:
                radius = int(round((w2 + h2)*0.25))
                frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
                eyeROI = frame_gray[y+y2:end_y, x+x2:end_x]
                cv.imshow("eye", eyeROI)

        for (x3,y3,w3,h3) in mouth:
            end_x, end_y = (x + x3 + w3, y + y3 + h3)
            if end_y < y + eyebrows_threshold_y * h:
                frame = cv.rectangle(frame, (x+x3,y+y3), (x+x3+w3,end_y), (0, 255, 255), 3)
                eyebrowROI = frame_gray[y+y3:end_y, x+x3:x+x3+w3]
                cv.imshow("eyebrow", eyebrowROI)

            elif end_y > end_h * mouth_threshold_y:
                frame = cv.rectangle(frame, (x+x3,y+y3), (x+x3+w3,end_y), (255, 255, 0), 3)
                mouthROI = frame_gray[y+y3:end_y, x+x3:x+x3+w3]
                cv.imshow("mouth", mouthROI)
                #cv.imwrite("lena_smile.png", mouthROI)

    cv.imshow('Live Emotion Recognition', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascade_eye.xml')
parser.add_argument('--mouth_cascade', help='Path to mouth cascade.', default='data/haarcascade_mcs_mouth.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
mouth_cascade_name = args.mouth_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
mouth_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

if not mouth_cascade.load(cv.samples.findFile(mouth_cascade_name)):
    print('--(!)Error loading mouth cascade')
    exit(0)

#-- 2. Read the video stream
camera_device = args.camera
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

image = cv.imread("lena_smile.png")
# detectAndDisplay(cv.imread("lena.jpg"))
auto = auto_canny(image)

cv.imshow("Original", image)
cv.imshow("Edges", auto)

cv.waitKey(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame)
#     if cv.waitKey(10) == 27:
#         break