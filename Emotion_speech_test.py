import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import csv
from PIL import Image, ImageTk
import speech_recognition as sr
import serial

# Constants
MODEL_PATH = 'emotion_recognition_model.h5'
LABEL_ENCODER_PATH = 'label_encoder_classes.npy'
DATA_PATH = 'TESS Toronto emotional speech set data'
HISTORY_FILE = 'prediction_history.csv'


def extract_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    return features


def train_and_save_model():
    data = []
    labels = []

    print("Starting feature extraction...")
    for folder_name in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder_name)
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            features = extract_features(file_path)
            data.append(features)
            labels.append(folder_name)

    print("Feature extraction complete. Preparing data...")
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    X_train = np.array(X_train)[:, np.newaxis, :]
    X_test = np.array(X_test)[:, np.newaxis, :]

    print("Building model...")
    model = Sequential()
    model.add(TimeDistributed(Dense(256, activation='relu'), input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy * 100:.2f}%')

    model.save(MODEL_PATH)
    np.save(LABEL_ENCODER_PATH, label_encoder.classes_)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder classes saved to {LABEL_ENCODER_PATH}")


class EmotionPredictor:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
        self.emotion_mapping = self._create_emotion_mapping()

    def _create_emotion_mapping(self):
        return {
            'YAF_angry': 'ANGRY', 'YAF_disgust': 'DISGUST', 'YAF_fear': 'FEAR',
            'YAF_happy': 'HAPPY', 'YAF_neutral': 'NEUTRAL', 'YAF_pleasant_surprised': 'SURPRISED', 'YAF_sad': 'SAD',
            'OAF_angry': 'ANGRY', 'OAF_disgust': 'DISGUST', 'OAF_Fear': 'FEAR',
            'OAF_happy': 'HAPPY', 'OAF_neutral': 'NEUTRAL', 'OAF_Pleasant_surprised': 'SURPRISED', 'OAF_Sad': 'SAD'
        }

    def predict_emotion(self, audio_file):
        stt_emotion = self._predict_from_speech(audio_file)
        if stt_emotion:
            return stt_emotion

        features = extract_features(audio_file)
        features = features[np.newaxis, np.newaxis, :]
        prediction = self.model.predict(features)
        index = np.argmax(prediction)
        emotion = self.label_encoder.classes_[index]
        return self.emotion_mapping.get(emotion, "UNKNOWN")

    def _predict_from_speech(self, audio_file):
        emotion_keywords = {
            "happy": "HAPPY", "sad": "SAD", "disgust": "DISGUST",
            "surprise": "SURPRISED", "angry": "ANGRY", "fear": "FEAR", "neutral": "NEUTRAL",
        }
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data).lower()
                for word, emotion in emotion_keywords.items():
                    if word in transcript:
                        return emotion
        except Exception as e:
            print(f"Speech-to-text failed: {e}")
        return None


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Prediction App")
        self.root.geometry("600x600")
        self.root.resizable(False, False)
        self.root.configure(bg='#F2F4F6')

        try:
            self.predictor = EmotionPredictor()
        except Exception as e:
            messagebox.showerror("Error", "Model load failed.")
            root.destroy()
            return

        self.emotion_to_emoji = {
            "HAPPY": "Emotion Emojis/happy.png",
            "SAD": "Emotion Emojis/sad.png",
            "ANGRY": "Emotion Emojis/angry.png",
            "SURPRISED": "Emotion Emojis/surprised.png",
            "NEUTRAL": "Emotion Emojis/neutral.png",
            "FEAR": "Emotion Emojis/fear.png",
            "DISGUST": "Emotion Emojis/disgust.png"
        }

        self.serial_port = None
        try:
            self.serial_port = serial.Serial('COM5', 9600, timeout=1)
        except Exception as e:
            print(f"Serial COM5 error: {e}")

        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "File Name", "Predicted Emotion"])

        self.build_ui()

    def build_ui(self):
        self.label = tk.Label(self.root, text="Upload Audio File", font=("Arial", 16), bg='#F2F4F6')
        self.label.pack(pady=20)

        self.upload_btn = tk.Button(self.root, text="Choose File", command=self.upload_audio, font=("Arial", 12))
        self.upload_btn.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 20, 'bold'), bg='#F2F4F6')
        self.result_label.pack(pady=20)

        self.emoji_label = tk.Label(self.root, bg='#F2F4F6')
        self.emoji_label.pack(pady=10)

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if file_path:
            emotion = self.predictor.predict_emotion(file_path)
            self.result_label.config(text=f"Emotion: {emotion}")

            if emotion in self.emotion_to_emoji:
                emoji_path = self.emotion_to_emoji[emotion]
                image = Image.open(emoji_path).resize((100, 100))
                self.emoji_image = ImageTk.PhotoImage(image)
                self.emoji_label.config(image=self.emoji_image)
            else:
                self.emoji_label.config(image='')

            self.save_prediction(file_path, emotion)
            self.send_serial(emotion)

    def save_prediction(self, file_name, emotion):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, os.path.basename(file_name), emotion])

    def send_serial(self, emotion):
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(emotion.encode())
                print(f"Sent to serial: {emotion}")
            except Exception as e:
                print(f"Error sending serial: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
