import dlib 
import imutils
import cv2

detec = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(detec)

cap = cv2.VideoCapture(0)  #  capturando imagem da webcam

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
       shape = predictor(gray, rect)
       for j in range (1, 68):
           cv2.putText(frame, str(j), (shape. part(j).x, shape.part(j).y), fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.3, color=(0,0,255)) 
           
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(5) & 0XFF
    if k ==27:
        break
cv2.destroyAllWindows()    
cap.release()

