# pip install anthropic,python-dotenv,sounddevice,numpy,wave,elevenlabs,openai-whisper,pynput,psutil,pillow

import os
import sys
import threading
import tempfile
import subprocess
import sounddevice as sd
import numpy as np
import wave
import anthropic
from dotenv import load_dotenv
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import whisper
from pynput import keyboard
import time
import psutil
import signal
import base64
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize Eleven Labs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Initialize Whisper model
whisper_model = whisper.load_model("base")

CHANNELS, RATE, CHUNK = 1, 16000, 1024

class VoiceAssistant:
    def __init__(self):
        self.is_recording = False
        self.is_playing = False
        self.running = True
        self.ctrl_pressed = self.shift_pressed = self.x_pressed = False
        self.recorded_frames = []
        self.conversation = []
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
            result = whisper_model.transcribe(filename)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def capture_screenshot(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            subprocess.run(['screencapture', '-x', temp_file.name], check=True)
            
            # Open the image using Pillow
            with Image.open(temp_file.name) as img:
                # Convert the image to RGB mode
                img = img.convert('RGB')
                
                # Calculate the aspect ratio
                aspect_ratio = img.width / img.height
                
                # Set the target width (you can adjust this)
                target_width = 1600
                
                # Calculate the new height based on the aspect ratio
                target_height = int(target_width / aspect_ratio)
                
                # Resize the image
                img_resized = img.resize((target_width, target_height), Image.LANCZOS)
                
                # Save the resized image to a BytesIO object
                buffer = BytesIO()
                img_resized.save(buffer, format="JPEG", quality=85, optimize=True)
                
                # Get the size of the compressed image
                buffer_size = buffer.getbuffer().nbytes
                print(f"Compressed image size: {buffer_size / 1024 / 1024:.2f} MB")
                
                # Encode the image to base64
                return base64.b64encode(buffer.getvalue()).decode()

    def get_ai_response(self, user_input):
        system_message = """You are a playful AI assistant. Be natural, use filler words,
        express emotions, but keep it concise, don't be chatty, use casual language, and adapt to the user's style.
        You have access to screenshots of the user's screen, but only describe or refer to
        the contents of these screenshots if the user specifically asks about them.
        Otherwise, focus on responding to the user's verbal input."""

        try:
            screenshot = self.capture_screenshot()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": screenshot
                            }
                        },
                        {
                            "type": "text",
                            "text": user_input
                        }
                    ]
                }
            ]

            if self.conversation:
                messages = self.conversation + messages

            print("Claude:", end=" ", flush=True)
            ai_response = ""
            with anthropic_client.messages.stream(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0.7,
                system=system_message,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    ai_response += text
            print()  # New line after the complete response
            
            self.conversation.append({"role": "user", "content": user_input})
            self.conversation.append({"role": "assistant", "content": ai_response})
            
            self.text_to_speech_and_play(ai_response)
        except Exception as e:
            print(f"AI response error: {e}")

    def text_to_speech_stream(self, text: str) -> IO[bytes]:
        """
        Converts text to speech and returns the audio data as a byte stream.
        """
        response = elevenlabs_client.text_to_speech.convert(
            voice_id="hprSi3ZqIjBZsmDs1j8D",  # Paola
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        print("Streaming audio data...")
        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)
        return audio_stream

    def text_to_speech_and_play(self, text):
        try:
            audio_stream = self.text_to_speech_stream(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
                temp_audio_file.write(audio_stream.getvalue())
                temp_audio_file.flush()
                
                self.is_playing = True
                self.playback_process = subprocess.Popen(["afplay", temp_audio_file.name])
                self.playback_process.wait()
                self.is_playing = False
                
            os.unlink(temp_audio_file.name)
        except Exception as e:
            print(f"Text-to-speech error: {e}")

def signal_handler(sig, frame):
    print("\nCtrl+C pressed. Exiting gracefully...")
    if assistant:
        assistant.running = False
    sys.exit(0)

def main():
    global assistant
    assistant = VoiceAssistant()
    signal.signal(signal.SIGINT, signal_handler)
    try:
        assistant.start()
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        print("Voice Assistant terminated.")

if __name__ == "__main__":
    main()