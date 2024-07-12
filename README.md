# 🎙️ Dekstop AI Voice Assistant

## 🌟 Introduction

Inspired by the ChatGPT Desktop App, I set out to build something similar: a desktop AI voice assistant that can see your screen and help with tasks. After two weeks of hundreds of chats with LLMs (mostly Claude 3.5 Sonnet) and trying out different solutions, this is what I finally put together.

I built three iterations of the final app to compare the performance of different LLMs and technologies:

1. 🟢 OpenAI Stack:
   - STT: Whisper API
   - LLM: gpt-4o
   - TTS: Whisper API

2. 🔵 Google Stack:
   - STT: Google Cloud API
   - LLM: Gemini 1.5 Flash
   - TTS: Google Cloud API

3. 🟣 Mixed solution:
   - STT: Whisper open source - local Python library
   - LLM: Claude 3.5 Sonnet
   - TTS: Eleven Labs API

Which one do you think would work the best? Try them out and let me know!

## 🚀 Features

- 🎤 Voice recording and transcription
- 🤖 Integration with AI models for conversation
- 🔊 Text-to-speech capabilities
- 📸 Screenshot capture for visual context
- ⌨️ Keyboard-driven interaction

## 📋 Prerequisites

- 🐍 Python 3.7+
- 🍎 An Apple Silicon Mac (scripts are tested on this platform)
- 💻 Visual Studio Code (recommended IDE)
- 🔑 API keys for the respective AI services (OpenAI, Anthropic, or Google AI)

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/voice-assistant-scripts.git
   cd voice-assistant-scripts
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages for each script:

   For the OpenAI script:
   ```
   pip install sounddevice soundfile numpy openai python-dotenv pynput
   ```

   For the Anthropic (Claude) script:
   ```
   pip install anthropic python-dotenv sounddevice numpy wave elevenlabs openai-whisper pynput psutil pillow
   ```

   For the Google AI script:
   ```
   pip install google-generativeai python-dotenv sounddevice numpy wave pynput google-cloud-speech google-cloud-texttospeech emoji psutil pillow
   ```

## ⚙️ Configuration

1. Create a `.env` file in the root directory of the project.
2. Add your API keys to the `.env` file:

   For OpenAI:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   For Anthropic (Claude):
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

   For Google AI:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_CLOUD_PROJECT=your_google_cloud_project_id_here
   ```

## 📝 Notes

- 🍎 These scripts are primarily tested on Apple Silicon Macs using Visual Studio Code.
- 🎤 Ensure you have the necessary permissions for microphone access and screen capture.
- 🤖 The scripts use different AI providers, so performance and capabilities may vary.
- 📜 Make sure to comply with the terms of service of the respective AI providers.

## 🔧 Troubleshooting

- 🔊 If you encounter audio-related issues, ensure that PortAudio is installed on your system.
- 🔑 For any API-related errors, double-check your API keys in the `.env` file.
- ☁️ Make sure you have the necessary Google Cloud credentials set up for the Google AI script.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is licensed under the MIT License.
