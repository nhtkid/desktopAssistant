# pip install sounddevice,soundfile,numpy,openai,python-dotenv,pynput

import os
import threading
import tempfile
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from openai import OpenAI
import subprocess
import base64
from dotenv import load_dotenv
import time
from pynput import keyboard
import logging
import signal

# Configure logging to only show WARNING and above
logging.basicConfig(level=logging.WARNING, format='%(message)s')

# Custom logger for user interactions
user_logger = logging.getLogger('user_interactions')
user_logger.setLevel(logging.INFO)
user_logger.propagate = False  # Prevent propagation to root logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
user_logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio recording parameters
CHANNELS = 1
RATE = 44100
MAX_RECORD_SECONDS = 30  # Maximum recording time

class VoiceAssistant:
    def __init__(self):
        self.frames = []
        self.is_recording = False
        self.is_speaking = False
        self.screenshot_path = None
        self.ctrl_pressed = False
        self.shift_pressed = False
        self.x_pressed = False
        self.running = True
        self.lock = threading.Lock()
        self.playback_process = None
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are a playful AI assistant. Be natural, use filler words,
                express emotions, but keep it concise, use casual language, and adapt to the user's style.
                You have access to screenshots of the user's screen, but only describe or refer to
                the contents of these screenshots if the user specifically asks about them.
                Otherwise, focus on responding to the user's verbal input."""
            }
        ]

    def start(self):
        print("Voice Assistant started. Press and hold Ctrl + Shift + X to speak. Press Esc to exit.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while self.running:
                time.sleep(0.1)

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            return False
        elif key == keyboard.Key.ctrl:
            self.ctrl_pressed = True
        elif key == keyboard.Key.shift:
            self.shift_pressed = True
        elif hasattr(key, 'char') and key.char == 'x':
            self.x_pressed = True
        
        if self.ctrl_pressed and self.shift_pressed and self.x_pressed and not self.is_recording:
            self.interrupt_playback()
            self.start_recording()

    def on_release(self, key):
        if key == keyboard.Key.ctrl:
            self.ctrl_pressed = False
        elif key == keyboard.Key.shift:
            self.shift_pressed = False
        elif hasattr(key, 'char') and key.char == 'x':
            self.x_pressed = False
        
        if not (self.ctrl_pressed and self.shift_pressed and self.x_pressed):
            if self.is_recording:
                self.stop_recording()

    def interrupt_playback(self):
        if self.playback_process and self.playback_process.poll() is None:
            os.killpg(os.getpgid(self.playback_process.pid), signal.SIGTERM)
            self.playback_process = None
        self.is_speaking = False

    def start_recording(self):
        with self.lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.frames = []
            print("Recording... (Release keys to stop)")
            threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
            print("Processing...")
            threading.Thread(target=self.process_audio).start()

    def record_audio(self):
        try:
            with sd.InputStream(channels=CHANNELS, samplerate=RATE, callback=self.audio_callback):
                sd.sleep(int(MAX_RECORD_SECONDS * 1000))
        except Exception as e:
            logging.error(f"Error during recording: {e}")

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self.frames.append(indata.copy())

    def process_audio(self):
        if not self.frames:
            logging.warning("No audio recorded.")
            return

        try:
            audio_data = np.concatenate(self.frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                sf.write(temp_audio.name, audio_data, RATE)
                transcript = self.transcribe_audio(temp_audio.name)
                if transcript.strip():
                    user_logger.info(f"You: {transcript}")
                    self.capture_screenshot()
                    self.get_ai_response(transcript)
                os.unlink(temp_audio.name)
        except Exception as e:
            logging.error(f"Error processing audio: {e}")

    def transcribe_audio(self, filename):
        try:
            with open(filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
                return transcription.text
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return ""

    def capture_screenshot(self):
        try:
            timestamp = int(time.time())
            self.screenshot_path = f"/tmp/screenshot_{timestamp}.png"
            result = subprocess.run(['screencapture', '-x', self.screenshot_path], check=True)
            if result.returncode != 0:
                logging.error("Failed to capture screenshot")
                self.screenshot_path = None
        except subprocess.CalledProcessError as e:
            logging.error(f"Error capturing screenshot: {e}")
            self.screenshot_path = None

    def encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return None

    def get_ai_response(self, user_input):
        user_message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_input},
            ]
        }

        if self.screenshot_path:
            base64_image = self.encode_image(self.screenshot_path)
            if base64_image:
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    }
                })

        self.conversation_history.append(user_message)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_history,
                max_tokens=150
            )

            ai_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            user_logger.info(f"AI: {ai_response}")
            self.text_to_speech(ai_response)

            # Clean up the temporary screenshot file
            if self.screenshot_path:
                os.unlink(self.screenshot_path)
                self.screenshot_path = None
        except Exception as e:
            logging.error(f"Error getting AI response: {e}")

    def text_to_speech(self, text):
        try:
            speech_file_path = Path(tempfile.gettempdir()) / "response.mp3"
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="nova",
                input=text
            )

            with open(speech_file_path, "wb") as f:
                f.write(response.content)

            self.play_audio(speech_file_path)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {e}")

    def play_audio(self, speech_file_path):
        self.is_speaking = True
        try:
            self.playback_process = subprocess.Popen(["afplay", str(speech_file_path)], preexec_fn=os.setsid)
            self.playback_process.wait()
        except subprocess.CalledProcessError as e:
            logging.error(f"Error playing audio: {e}")
        finally:
            self.is_speaking = False
            os.unlink(speech_file_path)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        assistant.start()
    except Exception as e:
        logging.critical(f"Critical error: {e}")
    finally:
        print("Voice Assistant terminated.")