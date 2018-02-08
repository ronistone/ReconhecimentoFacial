from register import RegisterUser
from recognition import Recognition
import os


while True:

    os.system("clear")
    print("1 - Cadastrar novo usuário")
    print("2 - Iniciar Reconhecimento")
    print("0 - Sair")

    o = input("Opção: ")

    if o == '0':
        break
    elif o == '1':
        RegisterUser('training-data')

    elif o == '2':
        rec = Recognition('training-data')
        rec.main_loop()
