import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)

DRAW_TARGET_FRAME = True
DRAW_HAND_LINES = False
DRAW_FINGER_CRACKS = True

     
def draw_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10,50), font, 2, (50, 50, 200), 3, cv2.LINE_AA)

while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
        # define range of skin color in HSV
        lower_skin = np.array([80,0,130], dtype=np.uint8)
        upper_skin = np.array([200,255,255], dtype=np.uint8)
        
        #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
        #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        
        
        #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
        #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
        #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
        #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
        #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
        #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
        #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        #l = no. of defects
        l=0
        
        #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)

            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [0, 0, 255], -1)
            
            if DRAW_HAND_LINES:
                cv2.line(roi,start, end, [0,255,0], 2)
            
            
        #print corresponding gestures which are in their ranges
        if l == 0:
            draw_text(frame, 'Rock')
        elif l == 1 or l == 2:
            draw_text(frame, 'Sxissors')
        elif l == 3 or l == 4:
            draw_text(frame, 'Paper')
            
        #show the windows
        cv2.imshow('mask',mask)
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

    
