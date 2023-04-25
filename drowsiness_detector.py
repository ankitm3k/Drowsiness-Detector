import cv2
import dlib
from scipy.spatial import distance
import concurrent.futures
import argparse
import time

def calculate_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_eye_aspect_ratio(leftEye)
        right_ear = calculate_eye_aspect_ratio(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR <= 0.25:
            cv2.putText(frame,"Warning:",(20,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            cv2.putText(frame,"Drowsiness Detection",(155,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(154,250,0),3)

    return frame

parser = argparse.ArgumentParser(description='Drowsiness Detection')
parser.add_argument('-v', '--video', type=str, default=None, help='Path to input video file')
parser.add_argument('-w', '--webcam', action='store_true', help='Use webcam for input')
args = parser.parse_args()

if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam!")
        exit()

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame!")
        break

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_frame, frame)
        output_frame = future.result()

    cv2.imshow("Drowsiness Detection", output_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()
