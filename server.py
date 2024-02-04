import socket, pickle, struct
import os
import face_recognition
import cv2
from safetensors import safe_open
print(cv2.__version__)

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

face_id_path = "face_id.safetensors"
if os.path.isfile(face_id_path):
    face_id = {}
    with safe_open(face_id_path, framework="np", device="cpu") as f:
        for key in f.keys():
            face_id[key] = f.get_tensor(key)
else:
    print('No file', face_id_path)
    raise Exception

# Socket Create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 9999
socket_address = ("192.168.1.35", port)

# Socket Bind
server_socket.bind(socket_address)

# Socket Listen
server_socket.listen(5)
print("LISTENING AT:", socket_address)

# Socket Accept
while True:
    client_socket, addr = server_socket.accept()
    print('GOT CONNECTION FROM:', addr)
    if client_socket:
        vid = cv2.VideoCapture(0)

        while (vid.isOpened()):
            img, frame = vid.read()
            copyImage = frame.copy()
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))
            for (x, y, w, h) in faces:
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                # coordinate = y - 10
                # cv2.rectangle(img, (x, coordinate), (x + 50, coordinate + 10), (255, 0, 0), 2)

                face_img = frame[y:y + h, x:x + w]  # Вырезаем лицо
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)  # Обязательно переводим в rgb
                face_encoded = face_recognition.face_encodings(face_img_rgb)  # Вычисляем код
                if len(face_encoded) > 0:
                    face_code = face_encoded[0]

                    results = face_recognition.compare_faces(list(face_id.values()), face_code)
                    names = [name for name, result in zip(face_id.keys(), results) if result]
                    print(names)
                cv2.rectangle(copyImage, (x, y), (x + w, y + h), (0, 255, 255), 15)
                if len(names) > 0:
                    name = str(names[0])
                else:
                    name = "Unknown"
                font = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 1
                thickness = 1
                textSize = cv2.getTextSize(name, font, fontScale, thickness)[0]
                text_width = textSize[0]
                text_height = textSize[1]

                cv2.rectangle(copyImage, (x, y + h - text_height - 10), (x + w, y + h), (0, 255, 255), 15)
                cv2.putText(copyImage, name, (x, y + h), font, 1, (255, 255, 255), 2)

            a = pickle.dumps(copyImage)
            message = struct.pack("Q", len(a)) + a
            client_socket.sendall(message)


            cv2.imshow('TRANSMITTING VIDEO', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                client_socket.close()