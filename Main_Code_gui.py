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
import serial.tools.list_ports
from transformers import pipeline
import torch
import cv2
import time
from threading import Thread

# Constants
AUDIO_MODEL_PATH = 'emotion_recognition_model.h5'
AUDIO_LABEL_ENCODER_PATH = 'label_encoder_classes.npy'
VISUAL_MODEL_PATH = 'model_file_30epochs.h5'
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATA_PATH = 'TESS Toronto emotional speech set data'
HISTORY_FILE = 'prediction_history.csv'


class EmotionSystem:
    def __init__(self):
        # Initialize serial connection
        self.serial_port = self.init_serial()

        # Initialize models
        self.audio_model = None
        self.audio_label_encoder = None
        self.text_model = None
        self.visual_model = None
        self.face_cascade = None

        self.load_audio_model()
        self.load_text_model()
        self.load_visual_model()

        # Emotion mappings
        self.emotion_mapping = {
            'YAF_angry': 'ANGRY', 'YAF_disgust': 'DISGUST', 'YAF_fear': 'FEAR',
            'YAF_happy': 'HAPPY', 'YAF_neutral': 'NEUTRAL', 'YAF_pleasant_surprised': 'SURPRISED', 'YAF_sad': 'SAD',
            'OAF_angry': 'ANGRY', 'OAF_disgust': 'DISGUST', 'OAF_Fear': 'FEAR',
            'OAF_happy': 'HAPPY', 'OAF_neutral': 'NEUTRAL', 'OAF_Pleasant_surprised': 'SURPRISED', 'OAF_Sad': 'SAD'
        }

        self.visual_labels = {
            0: 'ANGRY', 1: 'DISGUST', 2: 'FEAR',
            3: 'HAPPY', 4: 'NEUTRAL', 5: 'SAD', 6: 'SURPRISED'
        }

        self.emoji_paths = {
            "HAPPY": "Emotion Emojis/happy.png",
            "SAD": "Emotion Emojis/sad.png",
            "ANGRY": "Emotion Emojis/angry.png",
            "SURPRISED": "Emotion Emojis/surprised.png",
            "NEUTRAL": "Emotion Emojis/neutral.png",
            "FEAR": "Emotion Emojis/fear.png",
            "DISGUST": "Emotion Emojis/disgust.png"
        }

        # Emotion to serial code mapping
        self.emotion_codes = {
            "HAPPY": '1',
            "SAD": '2',
            "NEUTRAL": '3',
            "ANGRY": '4',
            "SURPRISED": '5',
            "FEAR": '6',
            "DISGUST": '7'
        }

        # Create history file if it doesn't exist
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Type", "Input", "Predicted Emotion", "Serial Command"])

    def init_serial(self):
        try:
            ports = serial.tools.list_ports.comports()
            available_ports = [port.device for port in ports]
            print("Available Ports:", available_ports)

            if 'COM4' in available_ports:
                s = serial.Serial('COM4', 9600, timeout=1)
                print("Serial connected to COM4.")
                return s
            else:
                print("COM4 not found. Serial not initialized.")
                return None
        except Exception as e:
            print(f"Serial connection error: {e}")
            return None

    def load_audio_model(self):
        try:
            self.audio_model = load_model(AUDIO_MODEL_PATH)
            self.audio_label_encoder = LabelEncoder()
            self.audio_label_encoder.classes_ = np.load(AUDIO_LABEL_ENCODER_PATH, allow_pickle=True)
            print("Audio model loaded successfully")
        except Exception as e:
            print(f"Audio model load failed: {e}")
            self.audio_model = None

    def load_text_model(self):
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.text_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=device
            )
            print("Text model loaded successfully")
        except Exception as e:
            print(f"Text model load failed: {e}")
            self.text_model = None

    def load_visual_model(self):
        try:
            self.visual_model = load_model(VISUAL_MODEL_PATH)
            self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            print("Visual model loaded successfully")
        except Exception as e:
            print(f"Visual model load failed: {e}")
            self.visual_model = None

    def extract_audio_features(self, file_path):
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        features = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
        return features

    def send_serial_command(self, emotion, gui_callback=None):
        """Send emotion code to Arduino and display status"""
        if not self.serial_port or not self.serial_port.is_open:
            status = "Serial port not available"
            if gui_callback:
                gui_callback(status)
            print(status)
            return False

        try:
            code = self.emotion_codes.get(emotion, '0')  # Default to 0 if unknown
            status = f"Sending to Arduino: {code} ({emotion})"
            if gui_callback:
                gui_callback(status)
            print(status)

            self.serial_port.write(f"{code}\n".encode())

            # Wait for acknowledgment
            start_time = time.time()
            while time.time() - start_time < 1.0:  # 1 second timeout
                if self.serial_port.in_waiting:
                    response = self.serial_port.readline().decode().strip()
                    status = f"Arduino response: {response}"
                    if gui_callback:
                        gui_callback(status)
                    print(status)
                    return True
                time.sleep(0.01)

            status = "No response from Arduino"
            if gui_callback:
                gui_callback(status)
            print(status)
            return False

        except Exception as e:
            status = f"Serial error: {e}"
            if gui_callback:
                gui_callback(status)
            print(status)
            return False

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

    def predict_audio_emotion(self, audio_file, gui_callback=None):
        # Try speech-to-text first
        stt_emotion = self._predict_from_speech(audio_file)
        if stt_emotion:
            self.send_serial_command(stt_emotion, gui_callback)
            return stt_emotion

        # If no speech detected or model fails, use audio features
        features = self.extract_audio_features(audio_file)
        features = features[np.newaxis, np.newaxis, :]
        prediction = self.audio_model.predict(features)
        index = np.argmax(prediction)
        emotion = self.audio_label_encoder.classes_[index]
        mapped_emotion = self.emotion_mapping.get(emotion, "UNKNOWN")
        self.send_serial_command(mapped_emotion, gui_callback)
        return mapped_emotion

    def predict_text_emotion(self, text, gui_callback=None):
        if not self.text_model:
            return None, None

        predictions = self.text_model(text)[0]
        predictions.sort(key=lambda x: x['score'], reverse=True)
        top_emotion = predictions[0]

        # Map 'joy' to 'happy' for consistency
        display_label = "HAPPY" if top_emotion['label'].strip().lower() == "joy" else top_emotion['label'].upper()

        self.send_serial_command(display_label, gui_callback)
        return display_label, predictions

    def predict_visual_emotion(self, frame, gui_callback=None):
        if not self.visual_model or not self.face_cascade:
            return frame, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = None

        for (x, y, w, h) in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))

            result = self.visual_model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            emotion = self.visual_labels[label]
            detected_emotion = emotion

            # Draw rectangle around face and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if detected_emotion:
            self.send_serial_command(detected_emotion, gui_callback)

        return frame, detected_emotion

    def save_prediction(self, input_type, input_data, emotion):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        code = self.emotion_codes.get(emotion, '0')
        with open(HISTORY_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            if input_type == "Audio":
                writer.writerow([timestamp, input_type, os.path.basename(input_data), emotion, code])
            elif input_type == "Text":
                # For text, store first 50 chars
                writer.writerow(
                    [timestamp, input_type, input_data[:50] + ("..." if len(input_data) > 50 else ""), emotion, code])
            else:  # Visual
                writer.writerow([timestamp, input_type, "Camera Frame", emotion, code])


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition System")
        self.root.geometry("900x800")

        # Custom color scheme
        self.bg_color = "#f0f2f5"
        self.header_color = "#2c3e50"
        self.button_color = "#3498db"
        self.button_hover = "#2980b9"
        self.success_color = "#2ecc71"
        self.warning_color = "#e74c3c"
        self.text_color = "#34495e"

        # Configure root window
        self.root.configure(bg=self.bg_color)
        style = ttk.Style()
        style.theme_use('clam')

        # Configure styles
        style.configure('TNotebook', background=self.bg_color)
        style.configure('TNotebook.Tab',
                        background="#bdc3c7",
                        padding=[15, 5],
                        font=('Arial', 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', self.header_color)],
                  foreground=[('selected', 'white')])

        # Initialize emotion system
        self.system = EmotionSystem()
        self.camera_active = False
        self.camera_thread = None

        # Create header frame
        self.header_frame = tk.Frame(self.root, bg=self.header_color, height=80)
        self.header_frame.pack(fill='x')

        # Header label
        self.header_label = tk.Label(self.header_frame,
                                     text="Emotion Recognition System",
                                     font=('Arial', 20, 'bold'),
                                     fg='white',
                                     bg=self.header_color)
        self.header_label.pack(pady=20)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # Create tabs
        self.audio_tab = ttk.Frame(self.notebook)
        self.text_tab = ttk.Frame(self.notebook)
        self.visual_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.audio_tab, text="Audio Analysis")
        self.notebook.add(self.text_tab, text="Text Analysis")
        self.notebook.add(self.visual_tab, text="Visual Analysis")
        self.notebook.add(self.history_tab, text="History")

        # Build each tab
        self.build_audio_tab()
        self.build_text_tab()
        self.build_visual_tab()
        self.build_history_tab()

        # Serial status frame
        self.serial_frame = tk.Frame(self.root, bg=self.bg_color)
        self.serial_frame.pack(fill='x', padx=10, pady=5)

        self.serial_status_var = tk.StringVar()
        self.serial_status_var.set("Serial: Ready")
        self.serial_status = tk.Label(self.serial_frame,
                                      textvariable=self.serial_status_var,
                                      font=('Arial', 10),
                                      bg=self.bg_color,
                                      fg=self.text_color)
        self.serial_status.pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = tk.Label(self.root,
                                   textvariable=self.status_var,
                                   bd=1,
                                   relief=tk.SUNKEN,
                                   anchor=tk.W,
                                   bg='#ecf0f1',
                                   fg=self.text_color,
                                   font=('Arial', 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready - Select a tab to begin analysis")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_button(self, parent, text, command):
        button = tk.Button(parent,
                           text=text,
                           command=command,
                           bg=self.button_color,
                           fg='white',
                           activebackground=self.button_hover,
                           activeforeground='white',
                           font=('Arial', 10, 'bold'),
                           relief=tk.FLAT,
                           padx=10,
                           pady=5,
                           borderwidth=0)
        button.bind("<Enter>", lambda e: button.config(bg=self.button_hover))
        button.bind("<Leave>", lambda e: button.config(bg=self.button_color))
        return button

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def update_serial_status(self, message):
        self.serial_status_var.set(f"Serial: {message}")
        self.root.update_idletasks()

    def build_audio_tab(self):
        tab = self.audio_tab
        tab.configure(style='TFrame')

        # Main content frame
        content_frame = tk.Frame(tab, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title frame
        title_frame = tk.Frame(content_frame, bg=self.bg_color)
        title_frame.pack(fill='x', pady=(0, 20))

        tk.Label(title_frame,
                 text="Audio Emotion Recognition",
                 font=('Arial', 16, 'bold'),
                 bg=self.bg_color,
                 fg=self.header_color).pack(side=tk.LEFT)

        # Model status
        model_status = "Loaded" if self.system.audio_model else "Not Available"
        status_color = self.success_color if self.system.audio_model else self.warning_color
        tk.Label(title_frame,
                 text=f"Audio Model: {model_status}",
                 font=('Arial', 10),
                 bg=self.bg_color,
                 fg=status_color).pack(side=tk.RIGHT, padx=10)

        # Upload button
        self.create_button(content_frame,
                           "Upload Audio File",
                           self.process_audio).pack(pady=20)

        # Result frame
        result_frame = tk.Frame(content_frame, bg=self.bg_color)
        result_frame.pack(fill='x', pady=20)

        self.audio_result_label = tk.Label(result_frame,
                                           text="",
                                           font=('Arial', 14),
                                           bg=self.bg_color,
                                           fg=self.text_color)
        self.audio_result_label.pack()

        self.audio_emoji_label = tk.Label(result_frame, bg=self.bg_color)
        self.audio_emoji_label.pack(pady=10)

    def build_text_tab(self):
        tab = self.text_tab
        tab.configure(style='TFrame')

        # Main content frame
        content_frame = tk.Frame(tab, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title frame
        title_frame = tk.Frame(content_frame, bg=self.bg_color)
        title_frame.pack(fill='x', pady=(0, 20))

        tk.Label(title_frame,
                 text="Text Emotion Recognition",
                 font=('Arial', 16, 'bold'),
                 bg=self.bg_color,
                 fg=self.header_color).pack(side=tk.LEFT)

        # Model status
        model_status = "Loaded" if self.system.text_model else "Not Available"
        status_color = self.success_color if self.system.text_model else self.warning_color
        tk.Label(title_frame,
                 text=f"Text Model: {model_status}",
                 font=('Arial', 10),
                 bg=self.bg_color,
                 fg=status_color).pack(side=tk.RIGHT, padx=10)

        # Input text
        tk.Label(content_frame,
                 text="Enter Text:",
                 font=('Arial', 12),
                 bg=self.bg_color,
                 fg=self.text_color).pack(pady=5)

        self.text_input = tk.Text(content_frame,
                                  height=8,
                                  width=60,
                                  font=('Arial', 11),
                                  wrap=tk.WORD)
        self.text_input.pack(pady=5)

        # Analyze button
        self.create_button(content_frame,
                           "Analyze Text",
                           self.process_text).pack(pady=10)

        # Results
        self.text_result_var = tk.StringVar()
        tk.Label(content_frame,
                 textvariable=self.text_result_var,
                 font=('Arial', 14, 'bold'),
                 bg=self.bg_color,
                 fg=self.success_color).pack(pady=5)

        # Detailed scores
        tk.Label(content_frame,
                 text="Emotion Scores:",
                 font=('Arial', 12),
                 bg=self.bg_color,
                 fg=self.text_color).pack(pady=5)

        self.scores_text = tk.Text(content_frame,
                                   height=6,
                                   width=60,
                                   state=tk.DISABLED,
                                   font=('Arial', 10),
                                   bg='white')
        self.scores_text.pack(pady=5)

    def build_visual_tab(self):
        tab = self.visual_tab
        tab.configure(style='TFrame')

        # Main content frame
        content_frame = tk.Frame(tab, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title frame
        title_frame = tk.Frame(content_frame, bg=self.bg_color)
        title_frame.pack(fill='x', pady=(0, 20))

        tk.Label(title_frame,
                 text="Visual Emotion Recognition",
                 font=('Arial', 16, 'bold'),
                 bg=self.bg_color,
                 fg=self.header_color).pack(side=tk.LEFT)

        # Model status
        model_status = "Loaded" if self.system.visual_model else "Not Available"
        status_color = self.success_color if self.system.visual_model else self.warning_color
        tk.Label(title_frame,
                 text=f"Visual Model: {model_status}",
                 font=('Arial', 10),
                 bg=self.bg_color,
                 fg=status_color).pack(side=tk.RIGHT, padx=10)

        # Camera feed
        self.camera_label = tk.Label(content_frame)
        self.camera_label.pack(pady=10)

        # Control buttons
        btn_frame = tk.Frame(content_frame, bg=self.bg_color)
        btn_frame.pack(pady=10)

        self.start_btn = self.create_button(btn_frame,
                                            "Start Camera",
                                            self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = self.create_button(btn_frame,
                                           "Stop Camera",
                                           self.stop_camera)
        self.stop_btn.config(bg='#95a5a6', activebackground='#7f8c8d')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.config(state=tk.DISABLED)

        # Result display
        self.visual_result_var = tk.StringVar()
        tk.Label(content_frame,
                 textvariable=self.visual_result_var,
                 font=('Arial', 14, 'bold'),
                 bg=self.bg_color,
                 fg=self.success_color).pack(pady=5)

        self.visual_emoji_label = tk.Label(content_frame, bg=self.bg_color)
        self.visual_emoji_label.pack(pady=10)

    def build_history_tab(self):
        tab = self.history_tab
        tab.configure(style='TFrame')

        # Main content frame
        content_frame = tk.Frame(tab, bg=self.bg_color)
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title frame
        title_frame = tk.Frame(content_frame, bg=self.bg_color)
        title_frame.pack(fill='x', pady=(0, 20))

        tk.Label(title_frame,
                 text="Prediction History",
                 font=('Arial', 16, 'bold'),
                 bg=self.bg_color,
                 fg=self.header_color).pack(side=tk.LEFT)

        # Treeview for history
        columns = ("Timestamp", "Type", "Input", "Emotion", "Serial Code")
        self.history_tree = ttk.Treeview(content_frame, columns=columns, show="headings")

        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120, anchor=tk.W)

        self.history_tree.column("Input", width=200)
        self.history_tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        # Load history
        self.load_history()

    def load_history(self):
        try:
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            # Read from file
            with open(HISTORY_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    self.history_tree.insert("", tk.END, values=row)
        except Exception as e:
            print(f"Error loading history: {e}")

    def process_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if not file_path:
            return

        self.update_status("Processing audio file...")

        try:
            emotion = self.system.predict_audio_emotion(file_path, self.update_serial_status)
            self.audio_result_label.config(text=f"Detected Emotion: {emotion}")

            # Display emoji if available
            if emotion in self.system.emoji_paths:
                try:
                    image = Image.open(self.system.emoji_paths[emotion]).resize((100, 100))
                    self.emoji_image = ImageTk.PhotoImage(image)
                    self.audio_emoji_label.config(image=self.emoji_image)
                except Exception as e:
                    print(f"Error loading emoji: {e}")
                    self.audio_emoji_label.config(image='')
            else:
                self.audio_emoji_label.config(image='')

            # Save to history
            self.system.save_prediction("Audio", file_path, emotion)
            self.update_status(f"Audio analysis complete: {emotion}")

            # Refresh history
            self.load_history()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process audio: {e}")
            self.update_status("Audio processing failed")

    def process_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Required", "Please enter some text.")
            return

        self.update_status("Analyzing text...")

        try:
            emotion, predictions = self.system.predict_text_emotion(text, self.update_serial_status)
            if not emotion:
                messagebox.showerror("Error", "Text model not available")
                return

            self.text_result_var.set(f"Top Emotion: {emotion}")

            # Display all scores
            detailed_scores = "\n".join([f"{p['label']}: {p['score']:.4f}" for p in predictions])
            self.scores_text.config(state=tk.NORMAL)
            self.scores_text.delete("1.0", tk.END)
            self.scores_text.insert(tk.END, detailed_scores)
            self.scores_text.config(state=tk.DISABLED)

            # Save to history
            self.system.save_prediction("Text", text, emotion)
            self.update_status(f"Text analysis complete: {emotion}")

            # Refresh history
            self.load_history()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze text: {e}")
            self.update_status("Text analysis failed")

    def start_camera(self):
        if self.camera_active:
            return

        self.camera_active = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("Starting camera...")

        # Start camera in a separate thread
        self.camera_thread = Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        self.camera_active = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Camera stopped")

    def camera_loop(self):
        cap = cv2.VideoCapture(0)
        last_emotion = None
        last_emotion_time = 0

        while self.camera_active:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process frame for emotion detection
                processed_frame, emotion = self.system.predict_visual_emotion(frame, self.update_serial_status)

                # Update display if we detected an emotion
                if emotion:
                    current_time = time.time()
                    # Only send if emotion changed or enough time passed
                    if emotion != last_emotion or (current_time - last_emotion_time) > 3:  # 3 seconds
                        last_emotion = emotion
                        last_emotion_time = current_time

                        # Save to history
                        self.system.save_prediction("Visual", "Camera Frame", emotion)
                        self.root.after(0, self.load_history)

                # Convert frame to PhotoImage
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)

                # Update GUI
                self.root.after(0, self.update_camera_display, img, emotion)

                # Small delay to prevent GUI freeze
                time.sleep(0.03)

            except Exception as e:
                print(f"Camera error: {e}")
                break

        cap.release()
        cv2.destroyAllWindows()

    def update_camera_display(self, img, emotion):
        self.camera_label.imgtk = img
        self.camera_label.configure(image=img)

        if emotion:
            self.visual_result_var.set(f"Detected Emotion: {emotion}")

            # Display emoji if available
            if emotion in self.system.emoji_paths:
                try:
                    emoji_img = Image.open(self.system.emoji_paths[emotion]).resize((100, 100))
                    self.emoji_photo = ImageTk.PhotoImage(emoji_img)
                    self.visual_emoji_label.config(image=self.emoji_photo)
                except Exception as e:
                    print(f"Error loading emoji: {e}")
                    self.visual_emoji_label.config(image='')
            else:
                self.visual_emoji_label.config(image='')

    def on_close(self):
        # Stop camera if running
        self.camera_active = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)

        # Close the main window
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()