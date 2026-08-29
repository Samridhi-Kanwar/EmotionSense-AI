import cv2
import numpy as np
from keras.models import load_model
import serial
import time

# Load the trained model
model = load_model('model_file_30epochs.h5')

# Try to connect to serial port
try:
    ser = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)  # wait for the serial connection to initialize
    print("Serial connection established on COM5")
except Exception as e:
    print("Serial connection failed:", e)
    ser = None

# Load face detection model
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Start video capture
video = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))

            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            emotion = labels_dict[label]

            print("Detected:", emotion)

            if emotion == 'Happy' and ser:
                try:
                    ser.write(b'1')
                    print("Sent '1' to COM5")  # confirmation
                except Exception as e:
                    print("Error writing to serial:", e)

            # Drawing bounding boxes and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as e:
        print("Error during frame processing:", e)

# Cleanup
video.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
