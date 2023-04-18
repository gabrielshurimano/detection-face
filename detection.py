import cv2
from playsound import playsound

# cria o objeto para capturar a imagem da webcam
cap = cv2.VideoCapture(0)

# cria o objeto de detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# variável para contar o número de rostos detectados
num_faces = 0

# loop para capturar a imagem da webcam e detectar rostos
while True:
    # captura a imagem da webcam
    ret, img = cap.read()
    
    # converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detecta rostos na imagem em escala de cinza
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # desenha um retângulo ao redor de cada rosto detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # atualiza o número de rostos detectados
    num_faces = len(faces)
    
    # verifica se há dois rostos detectados
    if num_faces == 3:
        # toca o som do alarme
        playsound("/home/gabriel/Documentos/scripts_estudos/detection_face/alarm.mp3")
    
    # exibe a imagem em tempo real
    cv2.imshow('img', img)
    
    # aguarda a tecla 'q' ser pressionada para encerrar o loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libera os recursos
cap.release()
cv2.destroyAllWindows()
