import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
import sounddevice as sd
import wave
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np

class VoiceRecorderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Voice Recorder")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        self.label = QLabel("Press 'Record' to start recording.")
        layout.addWidget(self.label)

        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.record_audio)
        layout.addWidget(self.record_button)

        self.save_button = QPushButton("Save and Process")
        self.save_button.clicked.connect(self.save_and_process_audio)
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.recording = None

    def record_audio(self):
        print("Recording started...")
        self.label.setText("Recording... Press 'Save and Process' when done.")
        self.record_button.setEnabled(False)
        self.save_button.setEnabled(False)
        duration = 5  # Record for 5 seconds
        self.recording = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
        sd.wait()
        self.label.setText("Recording complete.")
        self.save_button.setEnabled(True)

    def save_and_process_audio(self):
        self.label.setText("Saving and processing audio...")
        self.record_button.setEnabled(True)
        self.save_button.setEnabled(False)

        # Save the raw recording
        folder = "sample"
        os.makedirs(folder, exist_ok=True)
        raw_path = os.path.join(folder, "raw_audio.wav")
        self.save_wave_file(raw_path, self.recording)

        # Process audio to remove silence
        processed_path = os.path.join(folder, "processed_audio.wav")
        self.remove_silence(raw_path, processed_path)

        self.label.setText(f"Processed audio saved at: {processed_path}")

    def save_wave_file(self, filename, recording):
        with wave.open(filename, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(44100)
            wf.writeframes(recording.tobytes())

    def remove_silence(self, input_path, output_path):
        audio = AudioSegment.from_wav(input_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS - 14)
        processed_audio = sum((audio[start:end] for start, end in nonsilent_ranges), AudioSegment.silent(duration=0))
        processed_audio.export(output_path, format="wav")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    recorder = VoiceRecorderApp()
    recorder.show()
    sys.exit(app.exec_())