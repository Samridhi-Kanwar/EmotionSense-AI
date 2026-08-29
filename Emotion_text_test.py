import tkinter as tk
from tkinter import messagebox
from transformers import pipeline
import torch
import serial
import serial.tools.list_ports

# ----- SERIAL SETUP -----
def init_serial():
    try:
        ports = serial.tools.list_ports.comports()
        available_ports = [port.device for port in ports]
        print("🔌 Available Ports:", available_ports)

        if 'COM5' in available_ports:
            s = serial.Serial('COM5', 9600, timeout=1)
            print("✅ Serial connected to COM5.")
            return s
        else:
            print("❌ COM5 not found. Serial not initialized.")
            return None
    except Exception as e:
        print(f"❌ Serial connection error: {e}")
        return None

ser = init_serial()

# ----- DEVICE SETUP -----
device = 0 if torch.cuda.is_available() else -1
device_name = "CUDA (GPU)" if device == 0 else "CPU"

# ----- LOAD MODEL -----
print("📦 Loading model...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=device
)
print(f"✅ Model loaded on {device_name}")

# ----- FUNCTION TO PREDICT EMOTION -----
def predict_emotion():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", "Please enter some text.")
        return

    predictions = emotion_classifier(text)[0]
    predictions.sort(key=lambda x: x['score'], reverse=True)
    top_emotion = predictions[0]
    label_cleaned = top_emotion['label'].strip().lower()

    print(f"🔍 Top emotion label detected: '{label_cleaned}'")

    # Map 'joy' to 'happy' for consistency
    display_label = "Happy" if label_cleaned == "joy" else top_emotion['label'].capitalize()
    result_var.set(f"Top Emotion: {display_label} ({top_emotion['score']:.2f})")

    # Trigger serial if joy (aka happy)
    if label_cleaned == "joy":
        print("✅ Joy detected — sending serial as 'Happy'.")
        if ser:
            try:
                print("📤 Sending '1' to COM5...")
                #ser.write(b'1\n')
                #serial_status_var.set("✅ Serial sent: 1 (Happy detected via Joy)")
            except Exception as e:
                print(f"❌ Error writing to serial: {e}")
                #serial_status_var.set("❌ Failed to send to serial.")
        else:
            print("❌ Serial object is None.")
            serial_status_var.set("❌ Serial not initialized.")
    else:
        print(f"😐 Emotion is '{label_cleaned}', not 'joy/happy' — skipping serial.")
        #serial_status_var.set(f"⚠️ Serial not triggered (emotion ≠ Happy)")

    # Display all emotions
    detailed_scores = "\n".join([f"{p['label']}: {p['score']:.4f}" for p in predictions])
    scores_text.config(state=tk.NORMAL)
    scores_text.delete("1.0", tk.END)
    scores_text.insert(tk.END, detailed_scores)
    scores_text.config(state=tk.DISABLED)

# ----- TKINTER GUI -----
root = tk.Tk()
root.title("Emotion Detector (Text-based)")
root.geometry("420x460")
root.resizable(False, False)

tk.Label(root, text="Enter Text:", font=("Helvetica", 12)).pack(pady=5)
input_text = tk.Text(root, height=5, width=50)
input_text.pack(pady=5)

tk.Button(root, text="Detect Emotion", command=predict_emotion, bg="#4CAF50", fg="white").pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Helvetica", 14, "bold"), fg="blue").pack(pady=5)

tk.Label(root, text="All Emotions:", font=("Helvetica", 12)).pack(pady=5)
scores_text = tk.Text(root, height=7, width=50, state=tk.DISABLED)
scores_text.pack(pady=5)

serial_status_var = tk.StringVar()
tk.Label(root, textvariable=serial_status_var, font=("Helvetica", 10), fg="green").pack(pady=5)

tk.Label(root, text=f"Using device: {device_name}", font=("Helvetica", 10, "italic"), fg="gray").pack(pady=5)

root.mainloop()
