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
from PIL import Image, ImageTk, ImageDraw
import speech_recognition as sr
from transformers import pipeline
import torch
import cv2
import time
from threading import Thread
import json
import requests
import random
import math
import pyaudio
import wave
from collections import Counter

# Constants
AUDIO_MODEL_PATH = 'emotion_recognition_model.h5'
AUDIO_LABEL_ENCODER_PATH = 'label_encoder_classes.npy'
VISUAL_MODEL_PATH = 'model_file_30epochs.h5'
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATA_PATH = 'TESS Toronto emotional speech set data'
HISTORY_FILE = 'prediction_history.csv'
CONTENT_FILE = 'content_library.json'
CONFIG_FILE = 'app_config.json'

class EmotionSystem:
    def __init__(self):
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

        # Current emotions and confidence scores from different sources
        self.current_emotions = {
            "camera": None,
            "speech": None,
            "text": None
        }
        self.confidence_scores = {
            "camera": None,
            "speech": None,
            "text": None
        }

        # Workflow completion tracking
        self.workflow_completed = {
            "speech": False,
            "text": False,
            "camera": False
        }

        # Load configuration and content library
        self.app_config = self.load_app_config()
        self.content_library = self.load_content_library()

    def load_app_config(self):
        """Load application configuration from JSON file"""
        default_config = {
            "workflow_steps": [
                {
                    "id": "speech",
                    "name": "Speech Analysis",
                    "description": "Analyze emotion from audio/speech",
                    "required": False,
                    "order": 1
                },
                {
                    "id": "text", 
                    "name": "Text Analysis",
                    "description": "Analyze emotion from text input",
                    "required": False,
                    "order": 2
                },
                {
                    "id": "camera",
                    "name": "Camera Analysis", 
                    "description": "Analyze emotion from facial expressions",
                    "required": False,
                    "order": 3
                }
            ],
            "openai_config": {
                "enabled": True,
                "api_key": "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",
                "model": "gpt-3.5-turbo",
                "max_tokens": 150,
                "temperature": 0.7
            },
            "ui_config": {
                "theme": "default",
                "auto_generate": True,
                "show_workflow": True
            },
            "audio_config": {
                "record_duration": 5,
                "sample_rate": 16000,
                "channels": 1
            }
        }

        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    return self.merge_configs(default_config, loaded_config)
            else:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            print(f"Error loading app config: {e}")
            return default_config

    def merge_configs(self, default, loaded):
        """Recursively merge loaded config with default"""
        merged = default.copy()
        for key, value in loaded.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self.merge_configs(merged[key], value)
                else:
                    merged[key] = value
        return merged

    def load_content_library(self):
        """Load content from JSON file or create default if not exists"""
        try:
            if os.path.exists(CONTENT_FILE):
                with open(CONTENT_FILE, 'r') as f:
                    loaded_content = json.load(f)
                    print(f"DEBUG: Loaded content library with keys: {list(loaded_content.keys())[:5]}...")  # Show first 5 keys
                    return loaded_content
            else:
                # Generate all possible combinations as fallback
                emotions = ["HAPPY", "SAD", "ANGRY", "FEAR", "NEUTRAL", "SURPRISED", "DISGUST"]
                default_content = {}
                
                for e1 in emotions:
                    for e2 in emotions:
                        for e3 in emotions:
                            key = f"{e1}_{e2}_{e3}"
                            default_content[key] = [
                                f"Content for {e1}, {e2}, and {e3} emotions - this is a complex emotional state that deserves special attention.",
                                f"Understanding your mix of {e1}, {e2}, and {e3} - each emotion tells part of your story.",
                                f"Navigating {e1}, {e2}, and {e3} together requires emotional intelligence and self-awareness.",
                                f"Your emotional combination of {e1}, {e2}, and {e3} creates a unique psychological landscape.",
                                f"Embrace the complexity of feeling {e1}, {e2}, and {e3} simultaneously - it's what makes you human."
                            ]
                
                with open(CONTENT_FILE, 'w') as f:
                    json.dump(default_content, f, indent=2)
                print("DEBUG: Created default content library with all emotion combinations")
                return default_content
        except Exception as e:
            print(f"Error loading content library: {e}")
            # Return a minimal fallback
            return {"MIXED": ["Content not available for this emotion combination. Please check your content_library.json file."]}

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
                        return emotion, 0.85  # 85% confidence for keyword matching
        except Exception as e:
            print(f"Speech-to-text failed: {e}")
        return None, 0.0

    def predict_audio_emotion(self, audio_file):
        # Try speech-to-text first
        stt_emotion, stt_confidence = self._predict_from_speech(audio_file)
        if stt_emotion:
            self.current_emotions["speech"] = stt_emotion
            self.confidence_scores["speech"] = stt_confidence
            self.workflow_completed["speech"] = True
            return stt_emotion, stt_confidence

        # If no speech detected or model fails, use audio features
        if self.audio_model is None:
            return "NEUTRAL", 0.5

        features = self.extract_audio_features(audio_file)
        features = features[np.newaxis, np.newaxis, :]
        prediction = self.audio_model.predict(features)
        
        # Get confidence scores
        confidence = np.max(prediction)
        index = np.argmax(prediction)
        emotion = self.audio_label_encoder.classes_[index]
        mapped_emotion = self.emotion_mapping.get(emotion, "NEUTRAL")
        
        self.current_emotions["speech"] = mapped_emotion
        self.confidence_scores["speech"] = float(confidence)
        self.workflow_completed["speech"] = True
        
        return mapped_emotion, float(confidence)

    def predict_text_emotion(self, text):
        if not self.text_model:
            # Fallback: simple keyword matching
            emotion_keywords = {
                "happy": "HAPPY", "sad": "SAD", "angry": "ANGRY", 
                "fear": "FEAR", "surprise": "SURPRISED", "disgust": "DISGUST"
            }
            text_lower = text.lower()
            for word, emotion in emotion_keywords.items():
                if word in text_lower:
                    self.current_emotions["text"] = emotion
                    self.confidence_scores["text"] = 0.7
                    self.workflow_completed["text"] = True
                    return emotion, 0.7, []
            
            self.current_emotions["text"] = "NEUTRAL"
            self.confidence_scores["text"] = 0.6
            self.workflow_completed["text"] = True
            return "NEUTRAL", 0.6, []

        predictions = self.text_model(text)[0]
        predictions.sort(key=lambda x: x['score'], reverse=True)
        top_emotion = predictions[0]

        # Map 'joy' to 'happy' for consistency
        display_label = "HAPPY" if top_emotion['label'].strip().lower() == "joy" else top_emotion['label'].upper()
        
        self.current_emotions["text"] = display_label
        self.confidence_scores["text"] = top_emotion['score']
        self.workflow_completed["text"] = True
        
        return display_label, top_emotion['score'], predictions

    def predict_visual_emotion(self, frame):
        if not self.visual_model or not self.face_cascade:
            return frame, "NEUTRAL", 0.5

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        detected_emotion = "NEUTRAL"
        detected_confidence = 0.5

        for (x, y, w, h) in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))

            result = self.visual_model.predict(reshaped)
            confidence = np.max(result)
            label = np.argmax(result, axis=1)[0]
            emotion = self.visual_labels[label]
            detected_emotion = emotion
            detected_confidence = float(confidence)

            # Draw rectangle around face and label with confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame, detected_emotion, detected_confidence

    def generate_content(self):
        """Generate content based on combined emotions from all available sources"""
        emotions = self.current_emotions
        confidences = self.confidence_scores
        
        # Filter out None and WAITING values - only use actual detected emotions
        valid_emotions = {k: v for k, v in emotions.items() if v is not None and v != "WAITING"}
        
        if not valid_emotions:
            return "Please use at least one emotion detection method to generate content."
        
        print(f"DEBUG: Current emotions - Camera: {emotions['camera']}, Speech: {emotions['speech']}, Text: {emotions['text']}")
        print(f"DEBUG: Valid emotion values: {list(valid_emotions.values())}")
        
        # Get available emotion values (excluding WAITING)
        emotion_values = [emotions[source] for source in ["camera", "speech", "text"] 
                        if emotions[source] is not None and emotions[source] != "WAITING"]
        
        print(f"DEBUG: Valid emotion values (cleaned): {emotion_values}")
        
        if not emotion_values:
            return "No valid emotions detected. Please try again with different inputs."
        
        # Find the most common emotion among available sources
        emotion_counter = Counter(emotion_values)
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        most_common_count = emotion_counter.most_common(1)[0][1]
        
        print(f"DEBUG: Emotion counts: {dict(emotion_counter)}")
        print(f"DEBUG: Most common emotion: {most_common_emotion} (count: {most_common_count})")
        
        # Create emotion key based on available emotions
        if len(emotion_values) == 3:
            emotion_key = "_".join(emotion_values)
        elif len(emotion_values) == 2:
            # If we have two different emotions, use both plus the most common one
            if len(set(emotion_values)) == 2:
                emotion_key = "_".join(emotion_values + [most_common_emotion])
            else:  # Both are the same emotion
                emotion_key = "_".join([emotion_values[0]] * 3)
        else:  # Only one emotion available
            emotion_key = "_".join([emotion_values[0]] * 3)
        
        print(f"DEBUG: Looking for emotion key: {emotion_key}")
        
        # Initialize best_match with default
        best_match = "MIXED"
        content_options = self.content_library.get("MIXED", ["Content not available for this emotion combination."])
        
        # Try exact match first
        if emotion_key in self.content_library:
            content_options = self.content_library[emotion_key]
            best_match = emotion_key
            print(f"DEBUG: Found exact match for {emotion_key}")
        else:
            # Try to find the best matching category
            emotion_counts = emotion_counter
            
            print(f"DEBUG: Emotion counts (cleaned): {emotion_counts}")
            
            # Find if we have a dominant emotion
            max_count = max(emotion_counts.values()) if emotion_counts else 0
            if max_count >= 2:  # If we have at least 2 of the same emotion
                dominant_emotion = [e for e, count in emotion_counts.items() if count == max_count][0]
                best_match = "_".join([dominant_emotion] * 3)
                print(f"DEBUG: Dominant emotion {dominant_emotion}, trying match: {best_match}")
                
                if best_match in self.content_library:
                    content_options = self.content_library[best_match]
                    print(f"DEBUG: Using dominant emotion match: {best_match}")
                else:
                    # Try to find any combination that contains the dominant emotion
                    matching_keys = [key for key in self.content_library.keys() 
                                if dominant_emotion in key and key != "MIXED"]
                    if matching_keys:
                        best_match = matching_keys[0]
                        content_options = self.content_library[best_match]
                        print(f"DEBUG: Using partial match: {best_match}")
                    else:
                        # Use the actual emotion as fallback instead of HAPPY_NEUTRAL_SAD
                        best_match = "_".join([dominant_emotion] * 3)
                        content_options = [f"Content focusing on {dominant_emotion} emotion. This emotional state deserves attention and understanding."]
                        print(f"DEBUG: Using emotion-specific fallback: {best_match}")
            else:
                # For mixed emotions with no dominant, try to find closest match
                if len(emotion_values) >= 2:
                    # Try combinations with the available emotions
                    possible_keys = [
                        "_".join(sorted(emotion_values + [most_common_emotion])),
                        "_".join([most_common_emotion] * 3)
                    ]
                    
                    for key in possible_keys:
                        if key in self.content_library:
                            best_match = key
                            content_options = self.content_library[key]
                            print(f"DEBUG: Found mixed emotion match: {best_match}")
                            break
                    else:
                        best_match = "MIXED"
                        content_options = self.content_library.get(best_match, content_options)
                        print(f"DEBUG: Using MIXED as fallback for mixed emotions")
                else:
                    # Single emotion case should have been handled above
                    best_match = "_".join([emotion_values[0]] * 3)
                    content_options = [f"Content focusing on {emotion_values[0]} emotion. Your emotional experience is valid and important."]
                    print(f"DEBUG: Using single emotion fallback: {best_match}")
        
        # Select random content from available options
        selected_content = random.choice(content_options)
        print(f"DEBUG: Selected content from category: {best_match} - Content: {selected_content}")
        
        # Add source information
        sources_used = [f"{source.upper()}" for source, emotion in valid_emotions.items()]
        source_info = f"Based on analysis from: {', '.join(sources_used)}\n\n"
        
        return source_info + selected_content

    def reset_workflow(self):
        """Reset workflow completion status"""
        for key in self.workflow_completed:
            self.workflow_completed[key] = False
        for key in self.current_emotions:
            self.current_emotions[key] = None
        for key in self.confidence_scores:
            self.confidence_scores[key] = None

    def save_prediction(self, input_type, input_data, emotion, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(HISTORY_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            confidence_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            if input_type == "Audio":
                writer.writerow([timestamp, input_type, os.path.basename(input_data), emotion, confidence_str])
            elif input_type == "Text":
                writer.writerow([timestamp, input_type, input_data[:50] + ("..." if len(input_data) > 50 else ""), emotion, confidence_str])
            else:  # Visual
                writer.writerow([timestamp, input_type, "Camera Frame", emotion, confidence_str])

class ModernEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EmotionSense AI - Advanced Emotion Recognition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f0f23')
        
        # Initialize emotion system
        self.system = EmotionSystem()
        self.camera_active = False
        self.camera_thread = None
        self.camera_window = None
        self.capture_button = None
        
        # Audio recording variables
        self.recording = False
        self.audio_frames = []
        self.audio_stream = None
        self.audio = pyaudio.PyAudio()
        
        # Modern color scheme
        self.colors = {
            'bg': '#0f0f23',
            'card_bg': '#1a1a2e',
            'accent': '#00ff88',
            'accent_secondary': '#0088ff',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'warning': '#ff4444',
            'success': '#00ff88',
            'progress': '#0088ff'
        }
        
        # Animation variables
        self.animation_angle = 0
        
        self.setup_gui()
        self.start_animations()
        
    def setup_gui(self):
        # Header with gradient
        self.header = tk.Frame(self.root, bg=self.colors['bg'], height=120)
        self.header.pack(fill='x', padx=20, pady=10)
        
        # Animated title
        self.title_label = tk.Label(
            self.header,
            text="EmotionSense AI",
            font=('Arial', 32, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg']
        )
        self.title_label.pack(pady=10)
        
        self.subtitle_label = tk.Label(
            self.header,
            text="Advanced Multi-Modal Emotion Recognition System",
            font=('Arial', 14),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        self.subtitle_label.pack()
        
        # Add OpenAI status to header
        self.add_openai_status_display()
        
        # Main container with scrollable left panel
        self.main_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Workflow (with scrollbar)
        self.left_frame = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.left_frame.pack(side='left', fill='y')
        
        # Create a canvas for scrolling
        self.left_canvas = tk.Canvas(self.left_frame, bg=self.colors['bg'], highlightthickness=0, width=320)
        self.left_canvas.pack(side='left', fill='y')
        
        # Add scrollbar to left canvas
        self.left_scrollbar = ttk.Scrollbar(self.left_frame, orient='vertical', command=self.left_canvas.yview)
        self.left_scrollbar.pack(side='right', fill='y')
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        
        # Create scrollable frame inside canvas
        self.left_panel = tk.Frame(self.left_canvas, bg=self.colors['bg'])
        self.left_canvas_window = self.left_canvas.create_window((0, 0), window=self.left_panel, anchor='nw')
        
        # Right panel - Content
        self.right_panel = tk.Frame(self.main_container, bg=self.colors['bg'])
        self.right_panel.pack(side='right', fill='both', expand=True, padx=(20, 0))
        
        self.build_workflow_panel()
        self.build_content_panel()
        
        # Update scroll region
        def configure_scroll_region(event):
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        
        self.left_panel.bind("<Configure>", configure_scroll_region)
        self.left_canvas.bind("<Configure>", lambda e: self.left_canvas.itemconfig(self.left_canvas_window, width=e.width))
        
    def add_openai_status_display(self):
        """Add OpenAI status display to the header"""
        openai_enabled = self.system.app_config["openai_config"]["enabled"]
        
        status_color = self.colors['success'] if openai_enabled else self.colors['warning']
        status_text = "🤖 OpenAI: Enabled" if openai_enabled else "🤖 OpenAI: Disabled"
        
        # Create status label in header
        self.openai_status_label = tk.Label(
            self.header,
            text=status_text,
            font=('Arial', 10, 'bold'),
            fg=status_color,
            bg=self.colors['bg'],
            pady=5
        )
        self.openai_status_label.pack()
        
    def build_workflow_panel(self):
        # Workflow card - with reduced spacing
        workflow_card = tk.Frame(self.left_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        workflow_card.pack(fill='x', pady=(0, 5))  # Reduced pady
        
        tk.Label(
            workflow_card,
            text="Analysis Workflow",
            font=('Arial', 14, 'bold'),  # Slightly smaller font
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=8  # Reduced padding
        ).pack()
        
        # Progress visualization
        self.progress_canvas = tk.Canvas(
            workflow_card,
            width=150,  # Smaller canvas
            height=150,
            bg=self.colors['card_bg'],
            highlightthickness=0
        )
        self.progress_canvas.pack(pady=5)  # Reduced padding
        
        # Workflow steps with reduced spacing
        self.steps_frame = tk.Frame(workflow_card, bg=self.colors['card_bg'])
        self.steps_frame.pack(fill='x', padx=15, pady=5)  # Reduced padding
        
        self.workflow_steps = [
            {"id": "speech", "name": "🎤 Speech Analysis", "status": "pending"},
            {"id": "text", "name": "📝 Text Analysis", "status": "pending"},
            {"id": "camera", "name": "📷 Camera Analysis", "status": "pending"}
        ]
        
        self.step_widgets = {}
        for step in self.workflow_steps:
            step_frame = tk.Frame(self.steps_frame, bg=self.colors['card_bg'])
            step_frame.pack(fill='x', pady=2)  # Reduced spacing between steps
            
            # Status indicator
            status_canvas = tk.Canvas(step_frame, width=18, height=18, bg=self.colors['card_bg'], highlightthickness=0)
            status_canvas.pack(side='left', padx=(0, 8))
            
            # Step label
            label = tk.Label(
                step_frame,
                text=step["name"],
                font=('Arial', 10),  # Smaller font
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg']
            )
            label.pack(side='left', fill='x', expand=True)
            
            self.step_widgets[step["id"]] = {
                "canvas": status_canvas,
                "label": label,
                "status": "pending"
            }
            
        # Control buttons with reduced spacing
        btn_frame = tk.Frame(workflow_card, bg=self.colors['card_bg'])
        btn_frame.pack(fill='x', padx=15, pady=8)  # Reduced padding
        
        self.create_modern_button(
            btn_frame,
            "🔄 Reset Workflow",
            self.reset_workflow,
            self.colors['accent_secondary']
        ).pack(fill='x', pady=3)  # Reduced spacing
        
        self.create_modern_button(
            btn_frame,
            "🚀 Generate Content",
            self.generate_content,
            self.colors['accent']
        ).pack(fill='x', pady=3)  # Reduced spacing
        
        # Quick actions card with reduced spacing
        actions_card = tk.Frame(self.left_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        actions_card.pack(fill='x', pady=5)  # Reduced spacing
        
        tk.Label(
            actions_card,
            text="Quick Actions",
            font=('Arial', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=8  # Reduced padding
        ).pack()
        
        # Create a frame for action buttons with proper spacing
        action_buttons_frame = tk.Frame(actions_card, bg=self.colors['card_bg'])
        action_buttons_frame.pack(fill='x', padx=15, pady=8)  # Reduced padding
        
        action_buttons = [
            ("🎵 Analyze Audio File", self.analyze_audio),
            ("🎤 Record & Analyze Speech", self.record_speech),
            ("📄 Analyze Text Input", self.analyze_text),
            ("📸 Start Camera Analysis", self.start_camera),
            ("📊 View Analysis History", self.view_history)
        ]
        
        for text, command in action_buttons:
            button = self.create_modern_button(
                action_buttons_frame,
                text,
                command,
                self.colors['card_bg'],
                hover_color='#252540'
            )
            button.pack(fill='x', pady=6)  # Reduced spacing

    def build_content_panel(self):
        # Emotion display card
        emotion_card = tk.Frame(self.right_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        emotion_card.pack(fill='x', pady=(0, 10))
        
        tk.Label(
            emotion_card,
            text="Emotion Analysis Results",
            font=('Arial', 16, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=10
        ).pack()
        
        # Emotion indicators
        self.emotion_frame = tk.Frame(emotion_card, bg=self.colors['card_bg'])
        self.emotion_frame.pack(fill='x', padx=20, pady=10)
        
        self.emotion_widgets = {}
        emotions = ["speech", "text", "camera"]
        
        for i, emotion_type in enumerate(emotions):
            emotion_indicator = tk.Frame(self.emotion_frame, bg=self.colors['card_bg'])
            emotion_indicator.grid(row=0, column=i, padx=10, sticky='nsew')
            self.emotion_frame.grid_columnconfigure(i, weight=1)
            
            # Canvas for emotion visualization
            canvas = tk.Canvas(emotion_indicator, width=80, height=80, bg=self.colors['card_bg'], highlightthickness=0)
            canvas.pack(pady=5)
            
            # Emotion label
            label = tk.Label(
                emotion_indicator,
                text=emotion_type.upper(),
                font=('Arial', 10, 'bold'),
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg']
            )
            label.pack()
            
            # Confidence
            confidence_label = tk.Label(
                emotion_indicator,
                text="Confidence: --",
                font=('Arial', 8),
                fg=self.colors['text_secondary'],
                bg=self.colors['card_bg']
            )
            confidence_label.pack()
            
            self.emotion_widgets[emotion_type] = {
                "canvas": canvas,
                "label": label,
                "confidence": confidence_label,
                "emotion": None,
                "confidence_value": None
            }
        
        # Initialize with waiting state
        self.initialize_waiting_state()
        
        # Content display card
        content_card = tk.Frame(self.right_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        content_card.pack(fill='both', expand=True, pady=10)
        
        tk.Label(
            content_card,
            text="Generated Content",
            font=('Arial', 16, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=10
        ).pack()
        
        # Content display with scrollbar
        content_container = tk.Frame(content_card, bg=self.colors['card_bg'])
        content_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.content_text = tk.Text(
            content_container,
            height=15,
            wrap='word',
            font=('Arial', 12),
            bg='#252540',
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent'],
            selectbackground=self.colors['accent_secondary'],
            relief='flat',
            padx=15,
            pady=15
        )
        
        scrollbar = tk.Scrollbar(content_container, orient='vertical', command=self.content_text.yview)
        self.content_text.configure(yscrollcommand=scrollbar.set)
        
        self.content_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Insert welcome message
        welcome_msg = """🌟 Welcome to EmotionSense AI! 🌟

This advanced system analyzes emotions through:
• 🎤 Speech/audio input (file or recording)
• 📝 Written text content  
• 📷 Facial expressions via camera

Use ANY combination of analysis methods to generate personalized content!

Ready to begin your emotional journey?"""
        
        self.content_text.insert('1.0', welcome_msg)
        self.content_text.config(state='disabled')
        
    def initialize_waiting_state(self):
        """Initialize emotion displays with waiting state"""
        for emotion_type in self.emotion_widgets:
            self.update_emotion_display(emotion_type, "WAITING", 0.0)
            
    def create_modern_button(self, parent, text, command, bg_color, hover_color=None):
        if hover_color is None:
            hover_color = self.lighten_color(bg_color, 20)
            
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Arial', 11, 'bold'),
            bg=bg_color,
            fg=self.colors['text_primary'],
            activebackground=hover_color,
            activeforeground=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2'
        )
        
        # Bind hover effects
        btn.bind('<Enter>', lambda e: btn.config(bg=hover_color))
        btn.bind('<Leave>', lambda e: btn.config(bg=bg_color))
        
        return btn
        
    def lighten_color(self, color, percent):
        """Lighten a color by given percent"""
        try:
            color = color.lstrip('#')
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            light_rgb = tuple(min(255, int(c + (255 - c) * percent / 100)) for c in rgb)
            return f'#{light_rgb[0]:02x}{light_rgb[1]:02x}{light_rgb[2]:02x}'
        except:
            return color
            
    def update_progress_circle(self, progress):
        """Update the circular progress indicator"""
        self.progress_canvas.delete("all")
        
        center_x, center_y = 75, 75  # Adjusted for smaller canvas
        radius = 35
        start_angle = 90
        end_angle = 90 + (360 * progress)
        
        # Background circle
        self.progress_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline='#333355', width=6, fill=self.colors['card_bg']  # Thinner line
        )
        
        # Progress arc
        if progress > 0:
            self.progress_canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=start_angle, extent=end_angle - start_angle,
                outline=self.colors['progress'], width=6, style='arc'  # Thinner line
            )
        
        # Progress text
        self.progress_canvas.create_text(
            center_x, center_y,
            text=f"{int(progress * 100)}%",
            font=('Arial', 14, 'bold'),  # Smaller font
            fill=self.colors['text_primary']
        )
        
    def update_step_status(self, step_id, status):
        """Update workflow step status"""
        if step_id in self.step_widgets:
            widget = self.step_widgets[step_id]
            canvas = widget["canvas"]
            label = widget["label"]
            
            canvas.delete("all")
            
            if status == "completed":
                # Green checkmark
                canvas.create_oval(2, 2, 16, 16, fill=self.colors['success'], outline='')
                canvas.create_text(9, 9, text="✓", font=('Arial', 9, 'bold'), fill=self.colors['bg'])
                label.config(fg=self.colors['success'])
            elif status == "processing":
                # Orange loading
                canvas.create_oval(2, 2, 16, 16, fill=self.colors['accent_secondary'], outline='')
                canvas.create_text(9, 9, text="⟳", font=('Arial', 9, 'bold'), fill=self.colors['bg'])
                label.config(fg=self.colors['accent_secondary'])
            else:
                # Gray pending
                canvas.create_oval(2, 2, 16, 16, fill='#333355', outline='')
                canvas.create_text(9, 9, text="○", font=('Arial', 9, 'bold'), fill=self.colors['text_secondary'])
                label.config(fg=self.colors['text_secondary'])
                
            widget["status"] = status
            
    def update_emotion_display(self, emotion_type, emotion, confidence):
        """Update emotion visualization"""
        if emotion_type in self.emotion_widgets:
            widget = self.emotion_widgets[emotion_type]
            canvas = widget["canvas"]
            
            canvas.delete("all")
            
            # Emotion color mapping
            emotion_colors = {
                "HAPPY": "#00ff88",
                "SAD": "#0088ff", 
                "ANGRY": "#ff4444",
                "FEAR": "#ffaa00",
                "SURPRISED": "#ff00ff",
                "NEUTRAL": "#8888ff",
                "DISGUST": "#aa00ff",
                "WAITING": "#666666"  # Gray for waiting state
            }
            
            color = emotion_colors.get(emotion, "#666666")
            
            # Draw emotion circle
            center_x, center_y = 40, 40
            radius = 30
            
            # Background
            canvas.create_oval(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                fill='#252540', outline=''
            )
            
            if emotion == "WAITING":
                # Show question mark for waiting state
                canvas.create_text(
                    center_x, center_y,
                    text="?",
                    font=('Arial', 16, 'bold'),
                    fill='#666666'
                )
                widget["confidence"].config(text="Waiting...", fg=self.colors['text_secondary'])
            else:
                # Emotion fill based on confidence
                if confidence and confidence > 0:
                    fill_radius = int(radius * confidence)
                    canvas.create_oval(
                        center_x - fill_radius, center_y - fill_radius,
                        center_x + fill_radius, center_y + fill_radius,
                        fill=color, outline=''
                    )
                
                # Emotion text
                canvas.create_text(
                    center_x, center_y,
                    text=emotion[0] if emotion else "?",
                    font=('Arial', 16, 'bold'),
                    fill=self.colors['text_primary']
                )
                
                # Update confidence label
                if confidence and confidence > 0:
                    confidence_color = self.colors['success'] if confidence > 0.7 else self.colors['accent_secondary'] if confidence > 0.5 else self.colors['warning']
                    widget["confidence"].config(
                        text=f"Confidence: {confidence:.1%}",
                        fg=confidence_color
                    )
                else:
                    widget["confidence"].config(text="Confidence: --", fg=self.colors['text_secondary'])
                
            widget["emotion"] = emotion
            widget["confidence_value"] = confidence
            
    def start_animations(self):
        """Start background animations"""
        self.animate_title()
        
    def animate_title(self):
        """Animate the title with color cycling"""
        colors = ['#00ff88', '#0088ff', '#ff00ff', '#ffaa00']
        current_color = colors[int(self.animation_angle / 90) % len(colors)]
        
        self.title_label.config(fg=current_color)
        self.animation_angle = (self.animation_angle + 5) % 360
        
        self.root.after(100, self.animate_title)
        
    def reset_workflow(self):
        """Reset the workflow"""
        self.system.reset_workflow()
        
        for step_id in self.step_widgets:
            self.update_step_status(step_id, "pending")
            
        # Reset to waiting state instead of None
        for emotion_type in self.emotion_widgets:
            self.update_emotion_display(emotion_type, "WAITING", 0.0)
            
        self.update_progress_circle(0)  # This will show 0%
        
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', "🔄 Workflow reset! Ready to start new analysis.\n\nUse ANY combination of analysis methods to generate personalized content.")
        self.content_text.config(state='disabled')
        
    def generate_content(self):
        """Generate content based on available emotions - works with any combination"""
        # Check which analyses are completed
        completed_analyses = []
        for emotion_type, widget in self.emotion_widgets.items():
            if widget["emotion"] and widget["emotion"] != "WAITING":
                completed_analyses.append(emotion_type)
        
        if not completed_analyses:
            messagebox.showwarning(
                "No Analysis Data", 
                "Please complete at least one analysis method (speech, text, or camera) before generating content."
            )
            return
        
        # Update system emotions
        for emotion_type, widget in self.emotion_widgets.items():
            self.system.current_emotions[emotion_type] = widget["emotion"]
            self.system.confidence_scores[emotion_type] = widget["confidence_value"]
        
        # Generate content
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        
        status_msg = f"🔄 Generating personalized content based on {len(completed_analyses)} analysis method(s)...\n\n"
        status_msg += f"Methods used: {', '.join(completed_analyses).upper()}\n\n"
        
        self.content_text.insert('1.0', status_msg)
        self.content_text.see('end')
        self.root.update()
        
        # Generate content using the system
        content = self.system.generate_content()
        
        # Display the content
        self.content_text.delete('1.0', 'end')
        
        # Build emotion summary for completed analyses only
        emotion_summary = []
        for etype in completed_analyses:
            widget = self.emotion_widgets[etype]
            emotion_summary.append(f"• {etype.upper()}: {widget['emotion']} ({widget['confidence_value']:.0%})")
        
        formatted_content = f"""🎭 Emotional Analysis Complete!

📊 Analysis Results:
{chr(10).join(emotion_summary)}

🌟 Personalized Content Recommendation:

{content}

💡 Remember: {len(completed_analyses)} out of 3 analysis methods were used to create this personalized content.
Your emotional journey is unique and valid! 🌈"""
        
        self.content_text.insert('1.0', formatted_content)
        self.content_text.config(state='disabled')
        
    def analyze_audio(self):
        """Analyze audio file"""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.update_step_status("speech", "processing")
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', "🎵 Processing audio file...\n\nAnalyzing speech patterns and emotional content...")
        self.content_text.config(state='disabled')
        self.root.update()
        
        # Process in a separate thread to avoid GUI freezing
        def process_audio():
            try:
                emotion, confidence = self.system.predict_audio_emotion(file_path)
                self.system.save_prediction("Audio", file_path, emotion, confidence)
                
                self.root.after(0, lambda: self.complete_audio_analysis(emotion, confidence))
                
            except Exception as e:
                self.root.after(0, lambda: self.analysis_failed("speech", str(e)))
        
        Thread(target=process_audio, daemon=True).start()
        
    def record_speech(self):
        """Record and analyze speech"""
        recording_dialog = tk.Toplevel(self.root)
        recording_dialog.title("Record Speech")
        recording_dialog.geometry("400x300")
        recording_dialog.configure(bg=self.colors['bg'])
        recording_dialog.transient(self.root)
        recording_dialog.grab_set()
        
        tk.Label(
            recording_dialog,
            text="🎤 Speech Recording",
            font=('Arial', 16, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg'],
            pady=10
        ).pack()
        
        # Recording status
        status_label = tk.Label(
            recording_dialog,
            text="Click 'Start Recording' to begin",
            font=('Arial', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg'],
            pady=10
        )
        status_label.pack()
        
        # Countdown label
        countdown_label = tk.Label(
            recording_dialog,
            text="",
            font=('Arial', 24, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg'],
            pady=10
        )
        countdown_label.pack()
        
        # Button frame
        button_frame = tk.Frame(recording_dialog, bg=self.colors['bg'])
        button_frame.pack(pady=20)
        
        def start_recording():
            self.recording = True
            self.audio_frames = []
            
            # Configure audio stream
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.system.app_config["audio_config"]["channels"],
                rate=self.system.app_config["audio_config"]["sample_rate"],
                input=True,
                frames_per_buffer=1024
            )
            
            start_btn.config(state='disabled')
            stop_btn.config(state='normal')
            status_label.config(text="🔴 Recording... Speak now!", fg=self.colors['warning'])
            
            # Recording thread
            def record_audio():
                record_duration = self.system.app_config["audio_config"]["record_duration"]
                start_time = time.time()
                
                while self.recording and (time.time() - start_time) < record_duration:
                    data = stream.read(1024)
                    self.audio_frames.append(data)
                    
                    # Update countdown
                    remaining = record_duration - (time.time() - start_time)
                    countdown_label.config(text=f"{int(remaining)}s")
                    
                    time.sleep(0.01)
                
                stream.stop_stream()
                stream.close()
                
                if self.recording:  # If not manually stopped
                    self.root.after(0, save_recording)
            
            Thread(target=record_audio, daemon=True).start()
        
        def stop_recording():
            self.recording = False
            start_btn.config(state='normal')
            stop_btn.config(state='disabled')
            status_label.config(text="Recording stopped", fg=self.colors['text_secondary'])
            countdown_label.config(text="")
        
        def save_recording():
            stop_recording()
            
            # Save recorded audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = f"recorded_speech_{timestamp}.wav"
            
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(self.system.app_config["audio_config"]["channels"])
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.system.app_config["audio_config"]["sample_rate"])
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            
            recording_dialog.destroy()
            
            # Analyze the recorded audio
            self.update_step_status("speech", "processing")
            self.content_text.config(state='normal')
            self.content_text.delete('1.0', 'end')
            self.content_text.insert('1.0', "🎤 Analyzing recorded speech...\n\nProcessing emotional content from your recording...")
            self.content_text.config(state='disabled')
            self.root.update()
            
            def analyze_recorded_audio():
                try:
                    emotion, confidence = self.system.predict_audio_emotion(temp_file)
                    self.system.save_prediction("Audio", "Recorded Speech", emotion, confidence)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                    
                    self.root.after(0, lambda: self.complete_audio_analysis(emotion, confidence))
                    
                except Exception as e:
                    self.root.after(0, lambda: self.analysis_failed("speech", str(e)))
            
            Thread(target=analyze_recorded_audio, daemon=True).start()
        
        start_btn = self.create_modern_button(
            button_frame,
            "🎤 Start Recording",
            start_recording,
            self.colors['accent']
        )
        start_btn.pack(side='left', padx=5)
        
        stop_btn = self.create_modern_button(
            button_frame,
            "🛑 Stop Recording",
            stop_recording,
            self.colors['warning']
        )
        stop_btn.pack(side='left', padx=5)
        stop_btn.config(state='disabled')
        
        cancel_btn = self.create_modern_button(
            button_frame,
            "❌ Cancel",
            recording_dialog.destroy,
            self.colors['card_bg']
        )
        cancel_btn.pack(side='left', padx=5)
        
    def complete_audio_analysis(self, emotion, confidence):
        """Complete audio analysis"""
        self.update_emotion_display("speech", emotion, confidence)
        self.update_step_status("speech", "completed")
        self.update_progress_visualization()
        
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', f"✅ Speech Analysis Complete!\n\nDetected Emotion: {emotion}\nConfidence: {confidence:.1%}\n\nThis emotion has been recorded and will be used for content generation.")
        self.content_text.config(state='disabled')
        
    def analyze_text(self):
        """Analyze text input"""
        # Create text input dialog
        text_dialog = tk.Toplevel(self.root)
        text_dialog.title("Enter Text for Analysis")
        text_dialog.geometry("500x400")
        text_dialog.configure(bg=self.colors['bg'])
        text_dialog.transient(self.root)
        text_dialog.grab_set()
        
        tk.Label(
            text_dialog,
            text="Enter text to analyze for emotional content:",
            font=('Arial', 12, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg'],
            pady=10
        ).pack(padx=20, pady=5)
        
        text_area = tk.Text(
            text_dialog,
            height=15,
            width=60,
            font=('Arial', 11),
            bg='#252540',
            fg=self.colors['text_primary'],
            relief='flat',
            padx=10,
            pady=10
        )
        text_area.pack(padx=20, pady=10, fill='both', expand=True)
        
        def process_text():
            text = text_area.get('1.0', 'end').strip()
            if not text:
                messagebox.showwarning("Input Required", "Please enter some text to analyze.")
                return
                
            text_dialog.destroy()
            self.update_step_status("text", "processing")
            
            self.content_text.config(state='normal')
            self.content_text.delete('1.0', 'end')
            self.content_text.insert('1.0', "📝 Analyzing text content...\n\nProcessing linguistic patterns and emotional cues...")
            self.content_text.config(state='disabled')
            self.root.update()
            
            # Process in thread
            def analyze_text_thread():
                try:
                    emotion, confidence, predictions = self.system.predict_text_emotion(text)
                    self.system.save_prediction("Text", text, emotion, confidence)
                    self.root.after(0, lambda: self.complete_text_analysis(emotion, confidence))
                except Exception as e:
                    self.root.after(0, lambda: self.analysis_failed("text", str(e)))
            
            Thread(target=analyze_text_thread, daemon=True).start()
        
        btn_frame = tk.Frame(text_dialog, bg=self.colors['bg'])
        btn_frame.pack(padx=20, pady=10)
        
        self.create_modern_button(
            btn_frame,
            "🔍 Analyze Text",
            process_text,
            self.colors['accent']
        ).pack(side='left', padx=5)
        
        self.create_modern_button(
            btn_frame,
            "❌ Cancel",
            text_dialog.destroy,
            self.colors['warning']
        ).pack(side='left', padx=5)
        
    def complete_text_analysis(self, emotion, confidence):
        """Complete text analysis"""
        self.update_emotion_display("text", emotion, confidence)
        self.update_step_status("text", "completed")
        self.update_progress_visualization()
        
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', f"✅ Text Analysis Complete!\n\nDetected Emotion: {emotion}\nConfidence: {confidence:.1%}\n\nThis emotion has been recorded and will be used for content generation.")
        self.content_text.config(state='disabled')
        
    def start_camera(self):
        """Start camera analysis with popup window"""
        self.update_step_status("camera", "processing")
        
        # Create camera popup window
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Camera Emotion Analysis - Look at the Camera")
        self.camera_window.geometry("800x700")
        self.camera_window.configure(bg=self.colors['bg'])
        self.camera_window.transient(self.root)
        self.camera_window.grab_set()
        
        # Header
        tk.Label(
            self.camera_window,
            text="📷 Live Camera Emotion Analysis",
            font=('Arial', 18, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg'],
            pady=15
        ).pack()
        
        tk.Label(
            self.camera_window,
            text="Look directly at the camera and express your emotions naturally",
            font=('Arial', 12),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg'],
            pady=5
        ).pack()
        
        # Camera feed frame
        camera_frame = tk.Frame(self.camera_window, bg=self.colors['card_bg'], relief='ridge', bd=2)
        camera_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Camera display label
        self.camera_label = tk.Label(camera_frame, bg=self.colors['card_bg'])
        self.camera_label.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Control buttons frame
        control_frame = tk.Frame(self.camera_window, bg=self.colors['bg'])
        control_frame.pack(padx=20, pady=10, fill='x')
        
        # Capture Emotion button
        self.capture_button = self.create_modern_button(
            control_frame,
            "📸 Capture Current Emotion",
            self.capture_emotion,
            self.colors['accent']
        )
        self.capture_button.pack(side='left', padx=5)
        
        # Stop Camera button
        stop_button = self.create_modern_button(
            control_frame,
            "🛑 Stop Camera",
            self.stop_camera,
            self.colors['warning']
        )
        stop_button.pack(side='left', padx=5)
        
        # Status label
        self.camera_status = tk.Label(
            self.camera_window,
            text="Camera active - Detecting emotions...",
            font=('Arial', 10),
            fg=self.colors['success'],
            bg=self.colors['bg'],
            pady=5
        )
        self.camera_status.pack()
        
        # Current emotion display
        self.current_emotion_label = tk.Label(
            self.camera_window,
            text="Current Emotion: --",
            font=('Arial', 12, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['bg'],
            pady=5
        )
        self.current_emotion_label.pack()
        
        self.camera_confidence_label = tk.Label(
            self.camera_window,
            text="Confidence: --",
            font=('Arial', 10),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg']
        )
        self.camera_confidence_label.pack()
        
        # Start camera in a separate thread
        self.camera_active = True
        self.camera_thread = Thread(target=self.camera_analysis_loop, daemon=True)
        self.camera_thread.start()
        
        # Handle window close
        self.camera_window.protocol("WM_DELETE_WINDOW", self.stop_camera)
        
    def capture_emotion(self):
        """Capture the current emotion from camera"""
        if hasattr(self, 'last_emotion') and self.last_emotion:
            emotion = self.last_emotion
            confidence = self.last_confidence
            
            # Update system
            self.system.current_emotions["camera"] = emotion
            self.system.confidence_scores["camera"] = confidence
            self.system.workflow_completed["camera"] = True
            
            # Save prediction
            self.system.save_prediction("Visual", "Camera Frame", emotion, confidence)
            
            # Update GUI
            self.update_emotion_display("camera", emotion, confidence)
            self.update_step_status("camera", "completed")
            self.update_progress_visualization()
            
            # Show success message
            self.camera_status.config(
                text=f"✅ Emotion captured: {emotion} (Confidence: {confidence:.1%})",
                fg=self.colors['success']
            )
            
            # Update main content
            self.content_text.config(state='normal')
            self.content_text.delete('1.0', 'end')
            self.content_text.insert('1.0', f"✅ Camera Analysis Complete!\n\nDetected Emotion: {emotion}\nConfidence: {confidence:.1%}\n\nThis emotion has been recorded and will be used for content generation.")
            self.content_text.config(state='disabled')
            
            # Close camera window after a delay
            self.root.after(2000, self.stop_camera)
        else:
            self.camera_status.config(
                text="❌ No emotion detected yet. Please wait for detection...",
                fg=self.colors['warning']
            )
        
    def camera_analysis_loop(self):
        """Camera analysis loop with live feed"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame for emotion detection
                processed_frame, emotion, confidence = self.system.predict_visual_emotion(frame)
                
                # Store current emotion
                self.last_emotion = emotion
                self.last_confidence = confidence
                
                # Update camera window in main thread
                self.root.after(0, self.update_camera_display, processed_frame, emotion, confidence)
                
            except Exception as e:
                print(f"Camera error: {e}")
                break
                
            # Small delay to prevent high CPU usage
            time.sleep(0.03)
            
        cap.release()
        cv2.destroyAllWindows()
        
    def update_camera_display(self, frame, emotion, confidence):
        """Update camera display in the popup window"""
        if not self.camera_active or not hasattr(self, 'camera_label'):
            return
            
        # Convert frame to PhotoImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update camera label
        self.camera_label.img_tk = img_tk  # Keep reference
        self.camera_label.configure(image=img_tk)
        
        # Update emotion display
        if emotion and emotion != "NEUTRAL":
            self.current_emotion_label.config(
                text=f"Current Emotion: {emotion}",
                fg=self.colors['accent']
            )
            self.camera_confidence_label.config(
                text=f"Confidence: {confidence:.1%}",
                fg=self.colors['success'] if confidence > 0.7 else self.colors['accent_secondary']
            )
            
    def stop_camera(self):
        """Stop camera analysis"""
        self.camera_active = False
        if hasattr(self, 'camera_window') and self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
            
        if not self.system.workflow_completed["camera"]:
            self.update_step_status("camera", "pending")
            self.content_text.config(state='normal')
            self.content_text.delete('1.0', 'end')
            self.content_text.insert('1.0', "📷 Camera analysis stopped. You can restart it anytime.")
            self.content_text.config(state='disabled')
        
    def analysis_failed(self, step_id, error_msg):
        """Handle analysis failure"""
        self.update_step_status(step_id, "pending")
        
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', f"❌ Analysis Failed: {error_msg}\n\nPlease try again or check your input.")
        self.content_text.config(state='disabled')
        
    def view_history(self):
        """View analysis history"""
        try:
            if os.path.exists(HISTORY_FILE):
                os.startfile(HISTORY_FILE)
            else:
                messagebox.showinfo("History", "No analysis history found yet.")
        except:
            messagebox.showinfo("History", "History file is available at: " + os.path.abspath(HISTORY_FILE))
    
    def update_progress_visualization(self):
        """Update overall progress visualization"""
        completed_steps = sum(1 for widget in self.step_widgets.values() if widget["status"] == "completed")
        total_steps = len(self.step_widgets)
        progress = completed_steps / total_steps
        
        self.update_progress_circle(progress)

def main():
    root = tk.Tk()
    app = ModernEmotionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()