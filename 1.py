import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from x_ml import cal_angle_x
from y_ml import cal_angle_y

# Set the HD resolution (1280x720)
hd_width = 1280
hd_height = 720

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# Set the webcam resolution directly to 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, hd_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hd_height)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        # Index for a specific landmark (eye center in this case)
        es = [168]
        left_point = face[145]
        right_point = face[374]

        # Draw a circle at the specified landmarks
        for id in es:
            lm = face[id]  # Get the (x, y) coordinates of the landmark
            cv2.circle(img, lm, 3, (0, 0, 250), cv2.FILLED)

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

        print(f"{x_angle} || {y_angle}")

        # x1 = round((x_angle / 10),2)
        # y1 = round((y_angle / 10),2)
        # print(f"{x1} || {y1}")
                
        img = cv2.rectangle(img , (630,350) , (650,370) , (0,255,0) , 2 )
        img = cv2.circle(img , (640,360) , 3 , (255,0,0) , cv2.FILLED)
        # cvzone.putTextRect(img ,f'Depth: {int(depth1)} cms' ,(face[10][0] - 125,face[10][1]-40),
        #                     scale = 2) 


    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
