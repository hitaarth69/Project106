import cv2


fullbody_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
cap = cv2.VideoCapture('walking.avi')

while True:
    
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fullbodies = fullbody_classifier.detectMultiScale(grey, 1.2,3)
    for (x,y,w,h) in fullbodies:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),4)

        cv2.imshow("window",frame)

    if cv2.waitKey(1) == 32:
            break

    

   





    

    

cap.release()
cv2.destroyAllWindows()
