import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
#import RPi.GPIO as GPIO
from x_ml import cal_angle_x
from y_ml import cal_angle_y
from time import sleep
from scipy.spatial import distance


# Set the HD resolution (1280x720)----------------------------------------------

hd_width = 1280
hd_height = 720

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# Set the webcam resolution directly to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, hd_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hd_height)

#servo--------------------------------------------------------------------------

# GPIO setup
#GPIO.setmode(GPIO.BCM)

# Define GPIO pins
# SERVO1_PIN = 18
# SERVO2_PIN = 19

# GPIO.setup(SERVO1_PIN, GPIO.OUT)
# GPIO.setup(SERVO2_PIN, GPIO.OUT)

# servo1 = GPIO.PWM(SERVO1_PIN, 50)  # Servo 1
# servo2 = GPIO.PWM(SERVO2_PIN, 50)  # Servo 2

# Start PWM signals with 0% duty cycle
# servo1.start(0)
# servo2.start(0)


# drowsiness --------------------------------------------------------------------

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark
    C = distance.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
EYE_AR_THRESHOLD = 0.28
EYE_AR_CONSEC_FRAMES = 20

# Variables
COUNTER = 0

# Get the indexes for the left and right eyes (based on 68-point landmarks)
l_eye = [33, 160, 158, 133, 153, 144]
r_eye = [362, 385, 387, 263, 373, 380]

LEFT_EYE_POINTS = l_eye
RIGHT_EYE_POINTS = r_eye

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.flip(img, 1)

    img, faces = detector.findFaceMesh(img, draw=False)

# drowsiness---------------------------------------------------------------------------------------------------------------------------------------------------------

    for face in faces:
        
        # Extract the coordinates for the left and right eyes
        left_eye = [face[i] for i in LEFT_EYE_POINTS]
        right_eye = [face[i] for i in RIGHT_EYE_POINTS]


        # Compute the eye aspect ratio for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Visualize the eyes
        for (x, y) in left_eye + right_eye:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # Check if the eye aspect ratio is below the threshold
        if ear < EYE_AR_THRESHOLD:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

# main code-----------------------------------------------------------------------------------------------------------------------------------------------------------

    if faces:
        face = faces[0]

        # Index for a specific landmark (eye center in this case)
        es = [168]
        left_point = face[145]
        right_point = face[374]

        # Draw a circle at the specified landmarks
        for id in es:
            lm = face[id]  # Get the (x, y) coordinates of the landmark
            cv2.circle(img, lm, 3, (0, 255, 0), cv2.FILLED)

        #coordinates of the eye center
        x , y = lm

        # center of the window
        m = 640
        n = 360

        
        # calculating the distance between the driver and the camera
        def cal_depth():                                               
            w, _ = detector.findDistance(left_point , right_point)     #distance between the two points in pixels   
            W = 6.3    #------------------------------------------------average distance between the two points on the eyes

            #finding the distance from the driver to camera
            f = 1350              #focal length 
            d = (W * f) / w      #formula for calculating depth 
            return d

        #finding the distance to calculate the angle as well as calculating the angle
        depth1 = cal_depth()
        dist1 = x - m
        dist2 = y - n

        #print(f"{dist1} || {dist2}")
         
        pix1 = (dist1 / 2)
        pix2 = (dist2 / 2)

        #print(f"{pix1} || {pix2}")

        x_angle = cal_angle_x(pix1,depth1)
        y_angle = cal_angle_y(pix2,depth1)

        #print(f"{x_angle} || {y_angle}")

        # setting Servo 1 angle-------------------------------------------------------------------------------------------------------------------
        def set_servo1_angle(x_angle):
            duty_cycle = 2 + (x_angle / 18)   #(2-12%)
            #servo1.ChangeDutyCycle(duty_cycle)
            #sleep(0.3)  # Allow the servo to reach the position
            #servo1.ChangeDutyCycle(0)  # Stop signal to reduce jitter
            return duty_cycle

        # setting Servo 2 angle-------------------------------------------------------------------------------------------------------------------
        def set_servo2_angle(y_angle):    
            duty_cycle = 2 + (y_angle / 18)  # (2-12%)
            # servo2.ChangeDutyCycle(duty_cycle)
            # sleep(0.3)  # Allow the servo to reach the position
            # servo2.ChangeDutyCycle(0)  # Stop signal to reduce jitter
            return duty_cycle
        
        x_d_c = set_servo1_angle(x_angle)
        y_d_c = set_servo2_angle(y_angle)

        print(f"x_servo--{x_angle} || {x_d_c}") 
        print(f"y_servo--{y_angle} || {y_d_c}")
                        
        img = cv2.rectangle(img , (630,350) , (650,370) , (0,255,0) , 2 )
        img = cv2.circle(img , (640,360) , 3 , (255,0,0) , cv2.FILLED)
        img = cv2.line(img, (0,360), (1280,360), (0,0,255), 1)
        img = cv2.line(img, (640,0), (640,720), (0,0,255), 1) 


        # cvzone.putTextRect(img ,f'Depth: {int(depth1)} cms' ,(face[10][0] - 125,face[10][1]-40),
        #                     scale = 2) 


    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
