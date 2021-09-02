import cv2
import numpy as np
import math

SHOW_FRAME  = False
SHOW_MASK   = True

USE_DIALATION = True    
USE_GAUSSIAN_BLUR = True

DRAW_HAND_LINES         = True
DRAW_HULL_DEFECTS       = True
DRAW_FINGER_CRACKS      = True
DRAW_REGION_OF_INTEREST = True  

BLUE    = [255, 0, 0]
GREEN   = [0, 255, 0]
RED     = [0, 0, 255]

LOWER_SKIN_HSV = np.array([0,  17, 175], dtype=np.uint8)
UPPER_SKIN_HSV = np.array([28, 86, 255], dtype=np.uint8)

ROI_X_FROM = 0
ROI_X_TO = 300
ROI_Y_FROM = 0
ROI_Y_TO = 480

     
def draw_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10,50), font, 2, RED, 3, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
while(1):
    try:  
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        region_of_interest=frame[ROI_Y_FROM:ROI_Y_TO, ROI_X_FROM:ROI_X_TO]
        roi = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2HSV)

        if DRAW_REGION_OF_INTEREST:
            cv2.rectangle(frame, (ROI_X_FROM, ROI_Y_FROM), (ROI_X_TO, ROI_Y_TO), RED, 0)    

        # define range of skin color in HSV

        #extract skin colur image 
        mask = cv2.inRange(roi, LOWER_SKIN_HSV, UPPER_SKIN_HSV)
    
        #extrapolate the hand to fill dark spots within
        if USE_DIALATION:
            mask = cv2.dilate(mask, kernel, iterations = 3)
        
        
        #blur the image
        if USE_GAUSSIAN_BLUR:
            mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        #find edges
        contours, hierarchy= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #find edges of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        #approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        
        #make convex hull around hand
        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints = False)
        defects = cv2.convexityDefects(approx, hull)
        
        
        
        #l = no. of finger cracks
        l = 0
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            if DRAW_HULL_DEFECTS:
                cv2.circle(region_of_interest, far, 3, RED, -1)


            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
            
            #distance between point and convex hull
            d = (2 * ar) / a
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                if DRAW_FINGER_CRACKS:
                    cv2.circle(region_of_interest, far, 3, GREEN, -1)
            
            if DRAW_HAND_LINES:
                cv2.line(region_of_interest, start, end, BLUE, 2)
            
            
        #print corresponding gestures which are in their ranges
        if l == 0:
            draw_text(frame, 'Rock')
        elif l == 1 or l == 2:
            draw_text(frame, 'Scissors')
        elif l == 3 or l == 4:
            draw_text(frame, 'Paper')
            
        #show the windows
        if SHOW_MASK:
            cv2.imshow('mask',mask)
        if SHOW_FRAME:
            cv2.imshow('frame',frame)
        
    except Exception as e: 
        print('Error')
        print(e)
        #break
        pass
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    

    
