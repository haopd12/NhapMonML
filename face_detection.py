#Importing cv2
import cv2

#Loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame): 
    # We create a function that takes as input the image in black and white (gray) 
    #and the original image (frame), and that will return the same image with the detector rectangles. 
    
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    #scaleFactor--specifying how much the image size is reduced at each image scale
    #minNeighbors--specifying how many neighbors each candidate rectangle should have
    
    for (x, y, w, h) in faces: # For each detected face: (faces is the tuple of x,y--point of upper left corner,w-width,h-height)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 3)  #frame on which rectangle will be there,top-left,bottom-right,color,thickness
        
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image. (range from y to y+h)
        #This region is calculated as to save computation time to again search for eyes in whole image
        #It's better to detect a face and take the region of interest i.e. face and find eyes in it
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1,minNeighbors=22)
        
        for (ex, ey, ew, eh) in eyes: # For each detected eye: (Again retrieving x,y,w,h)
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 3)
        
    return frame # We return the image with the detector rectangles.  

