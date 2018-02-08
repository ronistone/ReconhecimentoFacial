import os
import cv2
import time
import numpy as np

class Recognition:

    def __init__(self, DEBUG=False):
        self.__DEBUG = DEBUG
        self.names = [""]
        self.path = 'training-data'
        self.setup()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.train(self.faces, np.array(self.labels))

        if self.__DEBUG:
            print("Total faces: ", len(self.faces))
            print("Total labels: ", len(self.labels))
            print("Pronto para reconhecimento")




    def detect_face(self,img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, minNeighbors=5,
                                                minSize=(20,20), maxSize=(400,400))

        if len(faces) == 0:
            return None,None

        (x,y,w,h) = faces[0]
        return gray[y:y+w, x:x+h], faces[0]


    def setup(self):
        dirs = os.listdir(self.path)
        self.faces = []
        self.labels = []

        print("Carregando... 0%")
        count = 1
        for dir_name in dirs:
            if not dir_name.startswith('user-'):
                continue
            print("Carregando... ",(count-1/len(dirs))*100,"%")
            count+=1
            self.names.append(dir_name.split('-')[1].replace("_"," "))
            label = int(dir_name.split('-')[2])

            subject_dir_path = self.path + '/' + dir_name+'/'
            images_names = os.listdir(subject_dir_path)

            for image_name in images_names:
                if image_name.startswith('.'):
                    continue

                image_path = subject_dir_path+"/"+image_name
                image = cv2.imread(image_path)

                if(self.__DEBUG):
                    cv2.imshow("Training on image...", image)
                    cv2.waitKey(50)

                face, rect = self.detect_face(image)

                if face is not None:
                    self.faces.append(face)
                    self.labels.append(label)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("Carregando... 100%")


    def predict(self, test_img):
        img = test_img.copy()
        face_test, rect = self.detect_face(img)

        if rect is None:
            return None, None

        label, confidence = self.recognizer.predict(face_test)
        label_text = self.names[label]

        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(img, self.names[label], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        return img, label

    def main_loop(self):

        webcam = cv2.VideoCapture(0)

        while True:
            s, image = webcam.read()
            image = cv2.flip(image, 180)
            test,id = self.predict(image)
            if id == None:
                test = image

            cv2.imshow("WebCAM", test)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rec = Recognition()
    rec.main_loop()
