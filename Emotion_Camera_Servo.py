import cv2
import numpy as np
from keras.models import load_model
import serial
import time
from datetime import datetime
import mysql.connector  # Import MySQL connector
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
import tempfile

# MySQL connection setup
db_config = {
    'host': 'localhost',
    'user': 'root',  # Update with your MySQL username
    'password': '',  # Update with your MySQL password
    'database': 'emotions_db'  # Name of your database
}

# Email setup
EMAIL_SENDER = 'theneerajjha@gmail.com'
EMAIL_PASSWORD = 'kbctnsmvtdptcuto'  # App-specific password
EMAIL_RECEIVER = 'theneerajjhaa@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# Function to send email with image attachment
def send_email(subject, body, image_path=None):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER

        msg.attach(MIMEText(body, 'plain'))

        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img:
                image_data = MIMEImage(img.read(), name=os.path.basename(image_path))
                msg.attach(image_data)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print("Email sent successfully.")
    except Exception as e:
        print("Email sending failed:", e)

# Function to log emotion into MySQL database
def log_emotion_to_mysql(emotion):
    try:
        # Establish database connection
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Insert emotion and timestamp into the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_query = "INSERT INTO emotion_log (timestamp, emotion) VALUES (%s, %s)"
        cursor.execute(insert_query, (timestamp, emotion))
        connection.commit()  # Commit the transaction
        print(f"Emotion '{emotion}' logged at {timestamp}")

    except mysql.connector.Error as err:
        print("Error logging emotion to database:", err)

    finally:
        # Close the cursor and connection
        if connection.is_connected():
            cursor.close()
            connection.close()

# Load the trained model
model = load_model('model_file_30epochs.h5')

# Try to connect to serial port
try:
    ser = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print("Serial connection established on COM4")
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

            # Log emotion with timestamp in the MySQL database
            log_emotion_to_mysql(emotion)

            # Save the detected face image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"face_{timestamp}.jpg"
            image_path = os.path.join(tempfile.gettempdir(), image_filename)
            cv2.imwrite(image_path, frame[y:y + h, x:x + w])

            # Send email with image if certain condition (e.g., 'Happy' emotion) is met
            if emotion == 'Happy':
                send_email(
                    "Emotion Detected",
                    f"Detected Emotion: {emotion} at {datetime.now().strftime('%H:%M:%S')}",
                    image_path=image_path
                )

            # Send signal to Arduino if the detected emotion is 'Happy'
            if emotion == 'Happy' and ser:
                try:
                    ser.write(b'1')
                    print("Sent '1' to COM5")
                except Exception as e:
                    print("Error writing to serial:", e)

            # Draw rectangle and label on the frame
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
