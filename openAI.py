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
import openai

# Constants
AUDIO_MODEL_PATH = 'emotion_recognition_model.h5'
AUDIO_LABEL_ENCODER_PATH = 'label_encoder_classes.npy'
VISUAL_MODEL_PATH = 'model_file_30epochs.h5'
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATA_PATH = 'TESS Toronto emotional speech set data'
HISTORY_FILE = 'prediction_history.csv'
CONTENT_FILE = 'content_library.json'
CONFIG_FILE = 'app_config.json'
OPENAI_LOG_FILE = 'openai_logs.json'

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

        # Initialize OpenAI client
        self.openai_client = None
        self.init_openai()

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

    def init_openai(self):
        """Initialize OpenAI client for old SDK (v0.x)"""
        try:
            import openai
            
            # Try to get API key from config first
            api_key = self.app_config.get("openai_config", {}).get("api_key", "") if hasattr(self, 'app_config') else ""
            
            # Fallback to environment variable
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")
            
            if api_key:
                openai.api_key = api_key
                self.openai_client = openai
                print(f"✅ OpenAI client initialized (SDK v{openai.__version__})")
                return True
            else:
                print("⚠️ OpenAI API key not found in config or environment")
                return False
        except Exception as e:
            print(f"❌ OpenAI initialization failed: {e}")
            return False

    def log_openai_request(self, request_data, response_data, error=None):
        """Log OpenAI API requests and responses for troubleshooting"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data if not error else None,
            "error": str(error) if error else None,
            "emotions": self.current_emotions.copy(),
            "confidences": self.confidence_scores.copy()
        }
        
        # Print to terminal for immediate troubleshooting
        print("\n" + "="*80)
        print("📝 OPENAI API REQUEST/RESPONSE LOG")
        print("="*80)
        print(f"📅 Timestamp: {log_entry['timestamp']}")
        print(f"🎭 Emotions: {log_entry['emotions']}")
        print(f"📊 Confidences: {log_entry['confidences']}")
        print("\n📤 REQUEST:")
        print(f"   Model: {request_data.get('model', 'N/A')}")
        print(f"   Max Tokens: {request_data.get('max_tokens', 'N/A')}")
        print(f"   Temperature: {request_data.get('temperature', 'N/A')}")
        print(f"   Messages: {request_data.get('messages', [])}")
        
        if error:
            print(f"\n❌ ERROR: {error}")
        else:
            print("\n📥 RESPONSE:")
            if isinstance(response_data, dict) and 'choices' in response_data:
                content = response_data['choices'][0]['message']['content']
                print(f"   Content: {content[:200]}..." if len(content) > 200 else f"   Content: {content}")
                print(f"   Finish Reason: {response_data['choices'][0]['finish_reason']}")
                print(f"   Usage: {response_data.get('usage', {})}")
            else:
                print(f"   Raw Response: {response_data}")
        print("="*80 + "\n")
        
        # Save to log file
        try:
            logs = []
            if os.path.exists(OPENAI_LOG_FILE):
                with open(OPENAI_LOG_FILE, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            # Keep only last 100 logs
            if len(logs) > 100:
                logs = logs[-100:]
            
            with open(OPENAI_LOG_FILE, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save OpenAI log: {e}")

    def generate_dynamic_content_openai(self, emotions, confidences):
        """Generate dynamic content using OpenAI API (old SDK)"""
        if not self.openai_client:
            print("⚠️ OpenAI client not initialized, using fallback content")
            return None
        
        try:
            # Prepare emotion summary
            emotion_summary = []
            for source, emotion in emotions.items():
                if emotion and emotion != "WAITING":
                    confidence = confidences.get(source, 0.0)
                    emotion_summary.append(f"{source.upper()}: {emotion} ({confidence:.0%} confidence)")
            
            if not emotion_summary:
                return None
            
            # Create prompt
            prompt = f"""Generate personalized emotional support content based on the following emotion analysis:
            
            Emotional Analysis Summary:
            {chr(10).join(emotion_summary)}
            
            The user is experiencing a combination of emotions. Provide:
            1. A brief analysis of what this emotional combination might mean
            2. Personalized advice or perspective
            3. A supportive message that validates their feelings
            4. One practical suggestion for emotional regulation
            
            Keep the tone warm, empathetic, and professional. Avoid generic phrases. Make it feel personalized."""
            
            # Get OpenAI config
            openai_config = self.app_config.get("openai_config", {})
            model = openai_config.get("model", "gpt-3.5-turbo")
            max_tokens = openai_config.get("max_tokens", 300)
            temperature = openai_config.get("temperature", 0.7)
            
            # Prepare request data for logging
            request_data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            print(f"\n🚀 Sending request to OpenAI API (Model: {model})...")
            
            # Make API call using old SDK
            response = self.openai_client.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Convert response to dict for logging
            response_dict = {
                "id": response.id,
                "model": response.model,
                "choices": [{
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    },
                    "finish_reason": choice.finish_reason
                } for choice in response.choices],
                "usage": response.usage if hasattr(response, 'usage') else {}
            }
            
            # Log the request/response
            self.log_openai_request(request_data, response_dict)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Add source attribution
            sources_used = [f"{source.upper()}" for source, emotion in emotions.items() 
                        if emotion and emotion != "WAITING"]
            source_info = f"✨ AI-Generated Content based on analysis from: {', '.join(sources_used)}\n\n"
            
            return source_info + content
            
        except Exception as e:
            error_msg = f"Error in OpenAI generation: {e}"
            print(f"❌ {error_msg}")
            self.log_openai_request(request_data if 'request_data' in locals() else {}, {}, error=e)
            return None
        except openai.RateLimitError as e:
            error_msg = f"OpenAI API rate limit exceeded: {e}"
            print(f"⏳ {error_msg}")
            self.log_openai_request(request_data, {}, error=e)
            return None
        except openai.APIError as e:
            error_msg = f"OpenAI API error: {e}"
            print(f"❌ {error_msg}")
            self.log_openai_request(request_data, {}, error=e)
            return None
        except Exception as e:
            error_msg = f"Unexpected error in OpenAI generation: {e}"
            print(f"❌ {error_msg}")
            self.log_openai_request(request_data if 'request_data' in locals() else {}, {}, error=e)
            return None

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
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "model": "gpt-3.5-turbo",
                "max_tokens": 300,
                "temperature": 0.7,
                "dynamic_content": True
            },
            "ui_config": {
                "theme": "default",
                "auto_generate": True,
                "show_workflow": True,
                "show_openai_status": True
            },
            "audio_config": {
                "record_duration": 5,
                "sample_rate": 16000,
                "channels": 1
            },
            "debug_config": {
                "log_openai_requests": True,
                "show_terminal_logs": True,
                "save_request_logs": True
            }
        }

        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    loaded_config = json.load(f)
                    merged_config = self.merge_configs(default_config, loaded_config)
                    
                    # Update OpenAI client if API key changed
                    if hasattr(self, 'openai_client'):
                        new_api_key = merged_config["openai_config"]["api_key"]
                        if new_api_key and (not openai.api_key or openai.api_key != new_api_key):
                            openai.api_key = new_api_key
                            self.openai_client = openai.OpenAI(api_key=new_api_key)
                            print("🔑 OpenAI API key updated from config")
                    
                    return merged_config
            else:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print("📁 Created new config file with default settings")
                return default_config
        except Exception as e:
            print(f"❌ Error loading app config: {e}")
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
                    print(f"📚 Loaded content library with {len(loaded_content)} entries")
                    return loaded_content
            else:
                # Generate minimal default content for fallback
                emotions = ["HAPPY", "SAD", "ANGRY", "FEAR", "NEUTRAL", "SURPRISED", "DISGUST"]
                default_content = {}
                
                # Create content for single emotions
                for emotion in emotions:
                    key = f"{emotion}_{emotion}_{emotion}"
                    default_content[key] = [
                        f"Focus on your {emotion.lower()} feelings. Acknowledge them without judgment.",
                        f"Your {emotion.lower()} emotion is valid. Take a moment to sit with these feelings.",
                        f"Experiencing {emotion.lower()} can be challenging. Remember, all emotions are temporary."
                    ]
                
                # Add some mixed emotion content
                default_content["MIXED"] = [
                    "You're experiencing a complex mix of emotions. This is completely normal and human.",
                    "Multiple emotions can coexist. Try to identify which feeling needs the most attention right now.",
                    "Your emotional landscape is rich and varied. Each emotion has something to teach you."
                ]
                
                with open(CONTENT_FILE, 'w') as f:
                    json.dump(default_content, f, indent=2)
                print(f"📝 Created default content library with {len(default_content)} entries")
                return default_content
        except Exception as e:
            print(f"❌ Error loading content library: {e}")
            return {"MIXED": ["Content not available. Please check your content_library.json file."]}

    def load_audio_model(self):
        try:
            self.audio_model = load_model(AUDIO_MODEL_PATH)
            self.audio_label_encoder = LabelEncoder()
            self.audio_label_encoder.classes_ = np.load(AUDIO_LABEL_ENCODER_PATH, allow_pickle=True)
            print("✅ Audio model loaded successfully")
        except Exception as e:
            print(f"❌ Audio model load failed: {e}")
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
            print("✅ Text model loaded successfully")
        except Exception as e:
            print(f"❌ Text model load failed: {e}")
            self.text_model = None

    def load_visual_model(self):
        try:
            self.visual_model = load_model(VISUAL_MODEL_PATH)
            self.face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
            print("✅ Visual model loaded successfully")
        except Exception as e:
            print(f"❌ Visual model load failed: {e}")
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
                        return emotion, 0.85
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
        
        # Filter out None and WAITING values
        valid_emotions = {k: v for k, v in emotions.items() if v is not None and v != "WAITING"}
        
        if not valid_emotions:
            return "Please use at least one emotion detection method to generate content."
        
        print(f"\n🎭 Generating content for emotions: {valid_emotions}")
        
        # Check if OpenAI dynamic content is enabled
        openai_config = self.app_config.get("openai_config", {})
        use_openai = openai_config.get("enabled", False) and openai_config.get("dynamic_content", True)
        
        if use_openai and self.openai_client:
            print("🤖 Attempting dynamic content generation with OpenAI...")
            dynamic_content = self.generate_dynamic_content_openai(valid_emotions, confidences)
            
            if dynamic_content:
                print("✅ Successfully generated dynamic content with OpenAI")
                return dynamic_content
            else:
                print("⚠️ OpenAI generation failed, falling back to static content")
        
        # Fallback to static content generation
        print("📄 Using static content library (OpenAI not available or failed)")
        return self.generate_static_content(valid_emotions)

    def generate_static_content(self, valid_emotions):
        """Generate static content from library (fallback method)"""
        emotion_values = list(valid_emotions.values())
        
        if not emotion_values:
            return "No valid emotions detected. Please try again with different inputs."
        
        # Find the most common emotion
        emotion_counter = Counter(emotion_values)
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        
        # Create emotion key
        if len(emotion_values) == 3:
            emotion_key = "_".join(emotion_values)
        elif len(emotion_values) == 2:
            if len(set(emotion_values)) == 2:
                emotion_key = "_".join(emotion_values + [most_common_emotion])
            else:
                emotion_key = "_".join([emotion_values[0]] * 3)
        else:
            emotion_key = "_".join([emotion_values[0]] * 3)
        
        # Try to get content from library
        content_options = self.content_library.get(emotion_key, None)
        
        if not content_options:
            # Try with dominant emotion
            dominant_key = "_".join([most_common_emotion] * 3)
            content_options = self.content_library.get(dominant_key, None)
        
        if not content_options:
            # Fallback to mixed content
            content_options = self.content_library.get("MIXED", ["Content not available for this emotion combination."])
        
        selected_content = random.choice(content_options)
        
        # Add source information
        sources_used = [f"{source.upper()}" for source in valid_emotions.keys()]
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
        
        # Add OpenAI settings button
        self.add_openai_settings_button()
        
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
        has_api_key = bool(self.system.app_config["openai_config"]["api_key"]) or bool(os.getenv("OPENAI_API_KEY"))
        
        if openai_enabled and has_api_key:
            status_color = self.colors['success']
            status_text = "🤖 OpenAI: Dynamic Content Enabled"
        elif openai_enabled and not has_api_key:
            status_color = self.colors['warning']
            status_text = "🤖 OpenAI: Enabled (No API Key)"
        else:
            status_color = self.colors['text_secondary']
            status_text = "🤖 OpenAI: Disabled"
        
        # Create status label in header
        self.openai_status_label = tk.Label(
            self.header,
            text=status_text,
            font=('Arial', 10, 'bold'),
            fg=status_color,
            bg=self.colors['bg'],
            pady=5,
            cursor='hand2'
        )
        self.openai_status_label.pack()
        self.openai_status_label.bind('<Button-1>', lambda e: self.show_openai_settings())

    def add_openai_settings_button(self):
        """Add OpenAI settings button to header"""
        settings_btn = tk.Button(
            self.header,
            text="⚙️ OpenAI Settings",
            font=('Arial', 9, 'bold'),
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            relief='flat',
            bd=0,
            padx=10,
            pady=5,
            cursor='hand2',
            command=self.show_openai_settings
        )
        settings_btn.pack(pady=5)

    def show_openai_settings(self):
        """Show OpenAI configuration dialog"""
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("OpenAI Configuration")
        settings_dialog.geometry("500x500")
        settings_dialog.configure(bg=self.colors['bg'])
        settings_dialog.transient(self.root)
        settings_dialog.grab_set()
        
        # Title
        tk.Label(
            settings_dialog,
            text="OpenAI API Configuration",
            font=('Arial', 16, 'bold'),
            fg=self.colors['accent'],
            bg=self.colors['bg'],
            pady=15
        ).pack()
        
        # Configuration frame
        config_frame = tk.Frame(settings_dialog, bg=self.colors['card_bg'], relief='ridge', bd=2, padx=20, pady=20)
        config_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Enable/disable checkbox
        self.openai_enabled_var = tk.BooleanVar(value=self.system.app_config["openai_config"]["enabled"])
        enabled_cb = tk.Checkbutton(
            config_frame,
            text="Enable OpenAI Dynamic Content",
            variable=self.openai_enabled_var,
            font=('Arial', 11, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            selectcolor=self.colors['card_bg'],
            activebackground=self.colors['card_bg'],
            activeforeground=self.colors['text_primary']
        )
        enabled_cb.grid(row=0, column=0, columnspan=2, sticky='w', pady=10)
        
        # API Key
        tk.Label(
            config_frame,
            text="API Key:",
            font=('Arial', 10),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg']
        ).grid(row=1, column=0, sticky='w', pady=5)
        
        api_key_var = tk.StringVar(value=self.system.app_config["openai_config"]["api_key"] or "")
        api_key_entry = tk.Entry(
            config_frame,
            textvariable=api_key_var,
            font=('Arial', 10),
            bg='#252540',
            fg=self.colors['text_primary'],
            insertbackground=self.colors['accent'],
            relief='flat',
            show="•"
        )
        api_key_entry.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Model selection
        tk.Label(
            config_frame,
            text="Model:",
            font=('Arial', 10),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg']
        ).grid(row=2, column=0, sticky='w', pady=5)
        
        model_var = tk.StringVar(value=self.system.app_config["openai_config"]["model"])
        model_combo = ttk.Combobox(
            config_frame,
            textvariable=model_var,
            values=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            font=('Arial', 10),
            state='readonly'
        )
        model_combo.grid(row=2, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Max tokens
        tk.Label(
            config_frame,
            text="Max Tokens:",
            font=('Arial', 10),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg']
        ).grid(row=3, column=0, sticky='w', pady=5)
        
        max_tokens_var = tk.IntVar(value=self.system.app_config["openai_config"]["max_tokens"])
        max_tokens_spin = tk.Spinbox(
            config_frame,
            from_=50,
            to=1000,
            increment=50,
            textvariable=max_tokens_var,
            font=('Arial', 10),
            bg='#252540',
            fg=self.colors['text_primary'],
            relief='flat'
        )
        max_tokens_spin.grid(row=3, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Temperature
        tk.Label(
            config_frame,
            text="Temperature:",
            font=('Arial', 10),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg']
        ).grid(row=4, column=0, sticky='w', pady=5)
        
        temp_var = tk.DoubleVar(value=self.system.app_config["openai_config"]["temperature"])
        temp_scale = tk.Scale(
            config_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient='horizontal',
            variable=temp_var,
            bg=self.colors['card_bg'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            length=200
        )
        temp_scale.grid(row=4, column=1, sticky='w', pady=5, padx=(10, 0))
        
        # Configure grid weights
        config_frame.grid_columnconfigure(1, weight=1)
        
        # Button frame
        btn_frame = tk.Frame(settings_dialog, bg=self.colors['bg'])
        btn_frame.pack(pady=20)
        
        def save_settings():
            # Update config
            self.system.app_config["openai_config"]["enabled"] = self.openai_enabled_var.get()
            self.system.app_config["openai_config"]["api_key"] = api_key_var.get()
            self.system.app_config["openai_config"]["model"] = model_var.get()
            self.system.app_config["openai_config"]["max_tokens"] = max_tokens_var.get()
            self.system.app_config["openai_config"]["temperature"] = temp_var.get()
            
            # Save to file
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(self.system.app_config, f, indent=4)
                
                # Reinitialize OpenAI client
                self.system.init_openai()
                
                # Update status display
                self.add_openai_status_display()
                
                messagebox.showinfo("Success", "OpenAI settings saved successfully!")
                settings_dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings: {e}")
        
        # Buttons
        self.create_modern_button(
            btn_frame,
            "💾 Save Settings",
            save_settings,
            self.colors['accent']
        ).pack(side='left', padx=5)
        
        self.create_modern_button(
            btn_frame,
            "❌ Cancel",
            settings_dialog.destroy,
            self.colors['warning']
        ).pack(side='left', padx=5)
        
        # View logs button
        self.create_modern_button(
            btn_frame,
            "📊 View API Logs",
            lambda: self.view_openai_logs(),
            self.colors['accent_secondary']
        ).pack(side='left', padx=5)

    def view_openai_logs(self):
        """View OpenAI API logs"""
        try:
            if os.path.exists(OPENAI_LOG_FILE):
                log_window = tk.Toplevel(self.root)
                log_window.title("OpenAI API Logs")
                log_window.geometry("900x600")
                log_window.configure(bg=self.colors['bg'])
                
                # Title
                tk.Label(
                    log_window,
                    text="OpenAI API Request/Response Logs",
                    font=('Arial', 16, 'bold'),
                    fg=self.colors['accent'],
                    bg=self.colors['bg'],
                    pady=10
                ).pack()
                
                # Text widget for logs
                text_frame = tk.Frame(log_window, bg=self.colors['bg'])
                text_frame.pack(fill='both', expand=True, padx=20, pady=10)
                
                log_text = tk.Text(
                    text_frame,
                    wrap='word',
                    font=('Courier', 9),
                    bg='#252540',
                    fg=self.colors['text_primary'],
                    insertbackground=self.colors['accent'],
                    relief='flat',
                    padx=10,
                    pady=10
                )
                
                scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=log_text.yview)
                log_text.configure(yscrollcommand=scrollbar.set)
                
                log_text.pack(side='left', fill='both', expand=True)
                scrollbar.pack(side='right', fill='y')
                
                # Load and display logs
                with open(OPENAI_LOG_FILE, 'r') as f:
                    logs = json.load(f)
                
                for i, log in enumerate(reversed(logs[-50:])):  # Show last 50 logs
                    log_text.insert('end', f"\n{'='*80}\n")
                    log_text.insert('end', f"Log Entry #{len(logs)-i}\n")
                    log_text.insert('end', f"Timestamp: {log.get('timestamp', 'N/A')}\n\n")
                    
                    log_text.insert('end', "Emotions: ")
                    log_text.insert('end', f"{log.get('emotions', {})}\n")
                    
                    if log.get('error'):
                        log_text.insert('end', f"\n❌ ERROR: {log['error']}\n")
                    else:
                        log_text.insert('end', "\n✅ Request successful\n")
                    
                    log_text.insert('end', "\n" + "="*80 + "\n")
                
                log_text.config(state='disabled')
                
                # Button frame
                btn_frame = tk.Frame(log_window, bg=self.colors['bg'])
                btn_frame.pack(pady=10)
                
                self.create_modern_button(
                    btn_frame,
                    "🗑️ Clear Logs",
                    lambda: self.clear_openai_logs(log_text),
                    self.colors['warning']
                ).pack(side='left', padx=5)
                
                self.create_modern_button(
                    btn_frame,
                    "📋 Copy to Clipboard",
                    lambda: self.copy_logs_to_clipboard(),
                    self.colors['accent_secondary']
                ).pack(side='left', padx=5)
                
                self.create_modern_button(
                    btn_frame,
                    "🚪 Close",
                    log_window.destroy,
                    self.colors['card_bg']
                ).pack(side='left', padx=5)
                
            else:
                messagebox.showinfo("No Logs", "No OpenAI API logs found yet.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load logs: {e}")

    def clear_openai_logs(self, log_text_widget=None):
        """Clear OpenAI logs"""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all OpenAI logs?"):
            try:
                with open(OPENAI_LOG_FILE, 'w') as f:
                    json.dump([], f)
                
                if log_text_widget:
                    log_text_widget.config(state='normal')
                    log_text_widget.delete('1.0', 'end')
                    log_text_widget.insert('end', "✅ Logs cleared successfully!")
                    log_text_widget.config(state='disabled')
                
                messagebox.showinfo("Success", "OpenAI logs cleared successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear logs: {e}")

    def copy_logs_to_clipboard(self):
        """Copy latest logs to clipboard"""
        try:
            if os.path.exists(OPENAI_LOG_FILE):
                with open(OPENAI_LOG_FILE, 'r') as f:
                    logs = json.load(f)
                
                # Format last 10 logs for clipboard
                clipboard_text = "OpenAI API Logs\n" + "="*50 + "\n\n"
                for i, log in enumerate(reversed(logs[-10:])):
                    clipboard_text += f"Log #{len(logs)-i} - {log.get('timestamp', 'N/A')}\n"
                    if log.get('error'):
                        clipboard_text += f"ERROR: {log['error']}\n"
                    else:
                        clipboard_text += f"Emotions: {log.get('emotions', {})}\n"
                    clipboard_text += "-"*30 + "\n"
                
                self.root.clipboard_clear()
                self.root.clipboard_append(clipboard_text)
                messagebox.showinfo("Success", "Last 10 logs copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy logs: {e}")

    def build_workflow_panel(self):
        # Workflow card - with reduced spacing
        workflow_card = tk.Frame(self.left_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        workflow_card.pack(fill='x', pady=(0, 5))
        
        tk.Label(
            workflow_card,
            text="Analysis Workflow",
            font=('Arial', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=8
        ).pack()
        
        # Progress visualization
        self.progress_canvas = tk.Canvas(
            workflow_card,
            width=150,
            height=150,
            bg=self.colors['card_bg'],
            highlightthickness=0
        )
        self.progress_canvas.pack(pady=5)
        
        # Workflow steps with reduced spacing
        self.steps_frame = tk.Frame(workflow_card, bg=self.colors['card_bg'])
        self.steps_frame.pack(fill='x', padx=15, pady=5)
        
        self.workflow_steps = [
            {"id": "speech", "name": "🎤 Speech Analysis", "status": "pending"},
            {"id": "text", "name": "📝 Text Analysis", "status": "pending"},
            {"id": "camera", "name": "📷 Camera Analysis", "status": "pending"}
        ]
        
        self.step_widgets = {}
        for step in self.workflow_steps:
            step_frame = tk.Frame(self.steps_frame, bg=self.colors['card_bg'])
            step_frame.pack(fill='x', pady=2)
            
            # Status indicator
            status_canvas = tk.Canvas(step_frame, width=18, height=18, bg=self.colors['card_bg'], highlightthickness=0)
            status_canvas.pack(side='left', padx=(0, 8))
            
            # Step label
            label = tk.Label(
                step_frame,
                text=step["name"],
                font=('Arial', 10),
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
        btn_frame.pack(fill='x', padx=15, pady=8)
        
        self.create_modern_button(
            btn_frame,
            "🔄 Reset Workflow",
            self.reset_workflow,
            self.colors['accent_secondary']
        ).pack(fill='x', pady=3)
        
        self.create_modern_button(
            btn_frame,
            "🚀 Generate Content",
            self.generate_content,
            self.colors['accent']
        ).pack(fill='x', pady=3)
        
        # Quick actions card with reduced spacing
        actions_card = tk.Frame(self.left_panel, bg=self.colors['card_bg'], relief='ridge', bd=2)
        actions_card.pack(fill='x', pady=5)
        
        tk.Label(
            actions_card,
            text="Quick Actions",
            font=('Arial', 14, 'bold'),
            fg=self.colors['text_primary'],
            bg=self.colors['card_bg'],
            pady=8
        ).pack()
        
        # Create a frame for action buttons with proper spacing
        action_buttons_frame = tk.Frame(actions_card, bg=self.colors['card_bg'])
        action_buttons_frame.pack(fill='x', padx=15, pady=8)
        
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
            button.pack(fill='x', pady=6)

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
        
        center_x, center_y = 75, 75
        radius = 35
        start_angle = 90
        end_angle = 90 + (360 * progress)
        
        # Background circle
        self.progress_canvas.create_oval(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            outline='#333355', width=6, fill=self.colors['card_bg']
        )
        
        # Progress arc
        if progress > 0:
            self.progress_canvas.create_arc(
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius,
                start=start_angle, extent=end_angle - start_angle,
                outline=self.colors['progress'], width=6, style='arc'
            )
        
        # Progress text
        self.progress_canvas.create_text(
            center_x, center_y,
            text=f"{int(progress * 100)}%",
            font=('Arial', 14, 'bold'),
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
                "WAITING": "#666666"
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
            
        self.update_progress_circle(0)
        
        self.content_text.config(state='normal')
        self.content_text.delete('1.0', 'end')
        self.content_text.insert('1.0', "🔄 Workflow reset! Ready to start new analysis.\n\nUse ANY combination of analysis methods to generate personalized content.")
        self.content_text.config(state='disabled')
        
    def generate_content(self):
        """Generate content based on available emotions"""
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
        
        # Check OpenAI status
        openai_enabled = self.system.app_config["openai_config"]["enabled"]
        has_api_key = bool(self.system.app_config["openai_config"]["api_key"]) or bool(os.getenv("OPENAI_API_KEY"))
        
        if openai_enabled and has_api_key and self.system.openai_client:
            status_msg = f"🚀 Generating AI-powered dynamic content based on {len(completed_analyses)} analysis method(s)...\n\n"
            status_msg += f"Methods used: {', '.join(completed_analyses).upper()}\n"
            status_msg += "🤖 Using OpenAI for dynamic content generation...\n\n"
        else:
            status_msg = f"📄 Generating content based on {len(completed_analyses)} analysis method(s)...\n\n"
            status_msg += f"Methods used: {', '.join(completed_analyses).upper()}\n"
            if openai_enabled and not has_api_key:
                status_msg += "⚠️ OpenAI enabled but no API key configured. Using static content.\n\n"
        
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
        
        # Add OpenAI attribution if used
        openai_used = openai_enabled and has_api_key and self.system.openai_client
        attribution = "✨ AI-Generated Content" if openai_used else "📚 Library Content"
        
        formatted_content = f"""🎭 Emotional Analysis Complete! {attribution}

📊 Analysis Results:
{chr(10).join(emotion_summary)}

🌟 Personalized Content Recommendation:

{content}

💡 Remember: {len(completed_analyses)} out of 3 analysis methods were used to create this content."""
        
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
    # Create necessary files and directories
    os.makedirs("Emotion Emojis", exist_ok=True)
    
    # Print startup info
    print("="*60)
    print("EmotionSense AI - Advanced Emotion Recognition System")
    print("="*60)
    print("\n📋 System Information:")
    print(f"   • OpenAI API Key Configured: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"   • TensorFlow Version: {tf.__version__}")
    print(f"   • OpenCV Version: {cv2.__version__}")
    print(f"   • OpenAI Version: {openai.__version__}")
    print("\n🚀 Starting application...")
    print("💡 Tip: Click '⚙️ OpenAI Settings' to configure your API key")
    print("="*60)
    
    root = tk.Tk()
    app = ModernEmotionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()