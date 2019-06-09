from tkinter import *

#Defining all Sub-routines:
####################################################################################
def identify():
    import cv2
    import numpy as np
    import os 
    identifier = cv2.face.LBPHFaceRecognizer_create()
    identifier.read('trainer/trainer.yml')
    cascadePath = "Cascades.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #font = FONT_HERSHEY_SCRIPT_SIMPLEX 
    #font =  FONT_ITALIC 

    #iniciate id counter
    id = 0
    # names related to ids respectively starting from zero
    names = ['None','Tahir','Safdar','Guest'] 
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = identifier.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less than 100 ==> "60" is perfect match 
            if (confidence < 70):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
        
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
###########################################################################################end recognizing

def entrant():
    import cv2
    import os
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    face_detector = cv2.CascadeClassifier('Cascades.xml')
    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            txtis= 'image saved: '+str(count)
            cv2.putText(img, txtis, (x+5,y-5), font, 1, (255,255,255), 2)
    
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
             break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


#######################################################################################end entry

def train():
    import cv2
    import numpy as np
    from PIL import Image
    import os
    # Path for face image database
    path = 'dataset'
    identifier = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades.xml")
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    identifier.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    identifier.write('trainer/trainer.yml') 
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    
#######################################################################################end training

main=Tk()
main.title("FYP GUI")
main.geometry("500x500")

#Adding a label
Label(main, text="Select").grid(row=0, column=0, sticky=W)

#Adding buttons
Button(main, text="Identify", width=14,bg="light green", command=identify).grid(row=1, column=0,sticky=W)
Button(main, text="New Entry", width=14,bg="sky blue", command=entrant).grid(row=2,  column=0 ,sticky=W)
Button(main, text="Training", width=14,bg="orange", command=train).grid(row=3,  column=0, sticky=W)
#calling the main function
main.mainloop()