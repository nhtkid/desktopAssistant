# pip install google-generativeai,python-dotenv,sounddevice,numpy,wave,pynput,google-cloud-speech,google-cloud-texttospeech,emoji,psutil,pillow

import os
import threading
import tempfile
import subprocess
import sounddevice as sd
import numpy as np
import wave
import google.generativeai as genai
from dotenv import load_dotenv
from pynput import keyboard
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.cloud import texttospeech
import time
import re
import emoji
import psutil
import uuid
from PIL import Image

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_API_KEY or not PROJECT_ID:
    print("Missing required environment variables")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
speech_client = SpeechClient()
tts_client = texttospeech.TextToSpeechClient()

CHANNELS, RATE, CHUNK = 1, 16000, 1024

def process_text_for_speech(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\b(haha|hehe)\b', '*laugh*', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(lol|lmao)\b', '*chuckle*', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*•\s*', 'Here\'s a point: ', text, flags=re.MULTILINE)
    text = re.sub(r'\((.*?)\)', r', \1,', text)
    text = re.sub(r'[^\w\s.,?!-]', '', text)
    return text.strip()

class VoiceAssistant:
    def __init__(self):
        self.is_recording = False
        self.is_playing = False
        self.running = True
        self.ctrl_pressed = self.shift_pressed = self.x_pressed = False
        self.recorded_frames = []
        self.model = genai.GenerativeModel('models/gemini-1.5-flash')
        self.chat = self.model.start_chat(history=[])
        self.chat.send_message("""You are a helpful AI assistant.""")
        self.playback_process = None

    def start(self):
        print("Voice Assistant started. Press and hold Ctrl+Shift+X to speak. Press Esc to exit.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            while self.running:
                time.sleep(0.1)

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
        elif key == keyboard.Key.ctrl: self.ctrl_pressed = True
        elif key == keyboard.Key.shift: self.shift_pressed = True
        elif hasattr(key, 'char') and key.char == 'x': self.x_pressed = True
        
        if self.ctrl_pressed and self.shift_pressed and self.x_pressed and not self.is_recording:
            self.interrupt_playback()
            self.start_recording()

    def on_release(self, key):
        if key == keyboard.Key.ctrl: self.ctrl_pressed = False
        elif key == keyboard.Key.shift: self.shift_pressed = False
        elif hasattr(key, 'char') and key.char == 'x': self.x_pressed = False
        
        if not (self.ctrl_pressed and self.shift_pressed and self.x_pressed) and self.is_recording:
            self.stop_recording()

    def interrupt_playback(self):
        if self.is_playing and self.playback_process:
            self.is_playing = False
            parent = psutil.Process(self.playback_process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            print("Playback interrupted.")

    def start_recording(self):
        print("Recording...")
        self.is_recording = True
        self.recorded_frames = []
        threading.Thread(target=self.record_audio).start()

    def stop_recording(self):
        self.is_recording = False
        threading.Thread(target=self.process_audio).start()

    def record_audio(self):
        def callback(indata, frames, time, status):
            if status: print(f"Audio callback status: {status}")
            self.recorded_frames.append(indata.copy())

        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
            while self.is_recording:
                sd.sleep(100)

    def process_audio(self):
        if not self.recorded_frames:
            print("No audio recorded.")
            return

        recording = np.concatenate(self.recorded_frames, axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            wf = wave.open(temp_audio.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes((recording * 32767).astype(np.int16).tobytes())
            wf.close()

            transcript = self.transcribe_audio(temp_audio.name)
            if transcript.strip():
                print(f"You: {transcript}")
                self.get_ai_response(transcript)

            os.unlink(temp_audio.name)

    def transcribe_audio(self, filename):
        try:
            with open(filename, "rb") as audio_file:
                content = audio_file.read()

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=["en-US"],
                model="long",
            )

            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
                config=config,
                content=content,
            )

            response = speech_client.recognize(request=request)

            return response.results[0].alternatives[0].transcript if response.results else ""
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def capture_screenshot(self):
        screenshot_path = f"/tmp/screenshot_{uuid.uuid4()}.png"
        subprocess.run(["screencapture", "-x", screenshot_path], check=True)
        return screenshot_path

    def get_ai_response(self, user_input):
        try:
            screenshot_path = self.capture_screenshot()
            image = Image.open(screenshot_path)
            
            response = self.chat.send_message([image, f"User said: {user_input}"], stream=True)
            
            ai_response = ""
            print("AI: ", end="", flush=True)
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    ai_response += chunk.text
            print()
            
            self.text_to_speech_and_play(ai_response)
            
            # Clean up
            os.unlink(screenshot_path)
        except Exception as e:
            print(f"AI response error: {e}")

    def text_to_speech_and_play(self, text):
        try:
            text = process_text_for_speech(text)
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US", 
                name="en-US-Journey-F"
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            response = tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_audio.write(response.audio_content)
                temp_audio_path = temp_audio.name

            self.is_playing = True
            self.playback_process = subprocess.Popen(["afplay", temp_audio_path])
            self.playback_process.wait()
            self.is_playing = False
            os.unlink(temp_audio_path)

        except Exception as e:
            print(f"Text-to-speech error: {e}")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        assistant.start()
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        print("Voice Assistant terminated.")