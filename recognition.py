import cv2
import numpy as np
import math

from numpy.core.records import array


SHOW_FRAME  = True
SHOW_MASK   = False

USE_DIALATION = True    
USE_GAUSSIAN_BLUR = True

DRAW_HAND_LINES         = True
DRAW_HULL_DEFECTS       = True
DRAW_FINGER_CRACKS      = True
DRAW_REGION_OF_INTEREST = True
DRAW_CONTOURS           = True  
DRAW_HULL               = True

BLUE    = [255, 0, 0]
GREEN   = [0, 255, 0]
RED     = [0, 0, 255]

LOWER_SKIN_HSV = np.array([154,  60, 130], dtype=np.uint8)
UPPER_SKIN_HSV = np.array([179, 255, 255], dtype=np.uint8)
KERNEL         = np.ones((3,3),np.uint8)

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
        _, frame = cap.read()
        frame=cv2.flip(frame,1)
        
        
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
            mask = cv2.dilate(mask, KERNEL, iterations = 3)
        
        
        #blur the image
        if USE_GAUSSIAN_BLUR:
            mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        #find edges
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #find edges of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        simple_cnt = cv2.approxPolyDP(cnt, 10, True)
        hull = cv2.convexHull(simple_cnt, returnPoints = False)
        hull_display = cv2.convexHull(simple_cnt, returnPoints = True)
        #approx the contour a little
        defects = cv2.convexityDefects(simple_cnt, hull)
        
        if DRAW_CONTOURS:
            #cv2.drawContours(frame, contours, -1, BLUE, 3)
            #cv2.drawContours(frame, [cnt], -1, RED, 3)
            cv2.drawContours(frame, [simple_cnt], -1, GREEN, 2)
        
        if DRAW_HULL:
            cv2.drawContours(frame, [hull_display], -1, GREEN, 4)

        #l = no. of finger cracks
        l = 0
        
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            hull_start_index, hull_end_index, farthest_point_index, distance = defects[i,0]
            
            hull_start_point    = tuple(simple_cnt[hull_start_index][0])
            hull_end_point      = tuple(simple_cnt[hull_end_index][0])
            farthest_point      = tuple(simple_cnt[farthest_point_index][0])
            
            l_hs_fp = math.sqrt((hull_start_point[0] - farthest_point[0]) ** 2 + (hull_start_point[1] - farthest_point[1]) ** 2) 
            l_he_fp = math.sqrt((hull_end_point[0]   - farthest_point[0]) ** 2 + (hull_end_point[1]   - farthest_point[1]) ** 2) 
            l_hs_he = math.sqrt((hull_start_point[0] - hull_end_point[0]) ** 2 + (hull_start_point[1] - hull_end_point[1]) ** 2) 

            right_ang_length = math.sqrt(l_he_fp ** 2 + l_hs_fp ** 2)

            if (right_ang_length < l_hs_he):
                continue

            l += 1
            if DRAW_FINGER_CRACKS:
                cv2.circle(region_of_interest, farthest_point, 3, BLUE, -1)
                cv2.circle(region_of_interest, hull_start_point, 3, RED, -1)
                cv2.circle(region_of_interest, hull_end_point, 3, RED, -1)
        
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

    
