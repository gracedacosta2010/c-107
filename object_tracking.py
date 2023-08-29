import cv2
import time
import math

video = cv2.VideoCapture("bb3.mp4")
tracker = cv2.TrackerCSRT_create
returned,img = video.read()
Bbox = cv2.selectROI("tracking",img,False)
tracker.init(img,Bbox)
print(Bbox)

def drawBox(img,Bbox):
    x,y,w,h = int(Bbox[0]),int(Bbox[1]),int(Bbox[2]),int(Bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img,"TRACKING",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)


while True:
    check,img = video.read()   
    success,Bbox = tracker.update(img)
    if success : 
        drawBox(img,Bbox)
    else :
        cv2.putText(img,"LOST",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)



    cv2.imshow("result",img)
            
    key = cv2.waitKey(25)

    if key == 32:
        print("Stopped!")
        break


video.release()
cv2.destroyALLwindows()



