import os
import cv2
import time
import numpy as np

class Recognition:

    def __init__(self,path, DEBUG=False):
        self.__DEBUG = DEBUG
        self.names = [""]
        self.path = path

        checkPath = self.path+'/check'
        if os.path.isfile(checkPath):
            with open(checkPath, 'r+') as f:
                if(f.read() == '1'):
                    self.isDirty = True
                else:
                    self.isDirty = False
        else:
            self.isDirty = True

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.isfile('train.yml'):
            self.recognizer.read('train.yml')

        self.setup()
        if self.isDirty:
            self.recognizer.train(self.faces, np.array(self.labels))
        self.recognizer.write("train.yml")
        with open(checkPath,'w') as f:
            f.write('0')

        if self.__DEBUG:
            print("Total faces: ", len(self.faces))
            print("Total labels: ", len(self.labels))
            print("Pronto para reconhecimento")

        self.faces = None
        self.labels = None



    def detect_face(self,img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')

        faces = face_cascade.detectMultiScale(gray, 1.2, minNeighbors=5,
                                                minSize=(10,10), maxSize=(600,600))

        if len(faces) == 0:
            return None,None

        return gray, faces


    def setup(self):
        dirs = os.listdir(self.path)
        self.faces = []
        self.labels = []

        count = 0
        print("Carregando... ",(count/len(dirs))*100,"%")
        for dir_name in dirs:
            if not dir_name.startswith('user-'):
                continue


            self.names.append(dir_name.split('-')[1].replace("_"," "))
            if self.isDirty:
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
                        (x,y,w,h) = rect[0]
                        self.faces.append(face[y:y+w, x:x+h])
                        self.labels.append(label)

            count+=1
            print("Carregando... ",(count/len(dirs))*100,"%")

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("Carregado")


    def predict(self, test_img):
        img = test_img.copy()
        gray, rect = self.detect_face(img)

        if rect is None:
            return None

        for (x,y,w,h) in rect:
            face_test = gray[y:y+w, x:x+h]
            label, confidence = self.recognizer.predict(face_test)
            if confidence < 40:
                label_text = self.names[label]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, self.names[label], (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, 'Nao Identificado', (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            print(confidence)

        return img

    def main_loop(self):

        webcam = cv2.VideoCapture(0)

        while True:
            s, image = webcam.read()
            image = cv2.flip(image, 180)
            test = self.predict(image)
            if test is None or test.all() == None:
                test = image

            cv2.imshow("WebCAM", test)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rec = Recognition('training-data', True)
    rec.main_loop()
