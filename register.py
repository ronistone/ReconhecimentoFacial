import cv2
import os
from time import sleep


class RegisterUser:

    def search_id(self):
        dirs = os.listdir(self.path)
        MAX = 0
        for dir_name in dirs:
            if not dir_name.startswith("user-"):
                continue
            MAX = max(MAX,int(dir_name.split('-')[2]))

        os.chdir(self.path)
        os.mkdir('user-'+self.name+"-"+str(MAX+1))
        os.chdir('..')
        return MAX+1,self.path+'/user-'+self.name+"-"+str(MAX+1)

    def takePhoto(self):

        arqCasc = 'lbpcascade_frontalface.xml'
        faceCascade = cv2.CascadeClassifier(arqCasc)

        webcam = cv2.VideoCapture(0)

        count = 1

        while True:
            s,image = webcam.read()
            cv2.imshow("Prepare-se e pressione a letra C quando estiver pronto...",cv2.flip(image,180))
            print("Prepare-se e pressione a letra C quando estiver pronto...")
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                break

        while True:
            s, image = webcam.read()
            image = cv2.flip(image,180)

            faces = faceCascade.detectMultiScale(
                        image,
                        1.1,
                        minNeighbors=5,
                        minSize=(20,20),
                        maxSize=(400,400))

            image1 = image
            if len(faces) > 0:
                (x,y,w,h) = faces[0]
                cv2.putText(image1, self.name, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0),2)
                cv2.rectangle(image1, (x,y+20), (x+w, y+h), (0,255,0),2)
                cv2.imshow('Salvando...', image1)
                print('Salvando Foto '+str(count))
                cv2.imwrite(self.path+'/image_'+str(count)+'.jpg',image)
                count += 1
                cv2.waitKey(1)
            else:
                print("face não reconhecida, tentando novamente...")
                cv2.imshow('Salvando...', image)
                cv2.waitKey(1)

            if count > 100:
                break

        cv2.destroyAllWindows()



    def __init__(self,path):
        os.system("clear")
        print("Cadastro de novo usuário\n")
        self.name = input("Nome: ")
        self.path = path
        self.id, self.path = self.search_id()

        print("Pasta criada, usuário id:",self.id)


        print("""\n\n\t\t\tAtenção:\n\n\tO sistema tirará fotos para o cadastro.
                O usuário a ser cadastrado deve permanecer frente a câmera\n""")
        input("Pressione ENTER para Continuar...")

        print(self.path)
        self.takePhoto()

        print("%s foi Cadastrado."%(self.name))


if __name__ == '__main__':
    RegisterUser('training-data')
