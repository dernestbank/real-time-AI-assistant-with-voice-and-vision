# app.py

import streamlit as st
import cv2
import time
import base64
from utils import WebcamStream, Assistant, setup_audio, load_configuration
import numpy as np

# Initialize session state
if 'webcam' not in st.session_state:
    st.session_state['webcam'] = WebcamStream().start()

if 'model' not in st.session_state:
    st.session_state['model'] = load_configuration()

if 'assistant' not in st.session_state:
    st.session_state['assistant'] = Assistant(st.session_state['model'])

if 'stop_listening' not in st.session_state:
    st.session_state['stop_listening'] = setup_audio(st.session_state['assistant'], st.session_state['webcam'])

# Streamlit layout
st.title("Live Video-Audio AI Communicating App")

# Placeholder for webcam
webcam_placeholder = st.empty()

# Placeholder for assistant responses
response_placeholder = st.empty()
# To allow users to type in prompts manually as an alternative to voice input.
st.markdown("### Send a Message")

user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        st.session_state['assistant'].answer(user_input, st.session_state['webcam'].read(encode=True))
        st.session_state['last_prompt'] = user_input

# a section to play the audio.
if hasattr(st.session_state['assistant'], 'last_audio') and st.session_state['assistant'].last_audio:
    audio_file = st.session_state['assistant'].last_audio
    # Convert PCM to WAV for Streamlit
    import wave

    def pcm_to_wav(pcm_file):
        wav_file = pcm_file.replace(".pcm", ".wav")
        with open(pcm_file, 'rb') as pcm, wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # paInt16
            wav.setframerate(24000)
            wav.writeframes(pcm.read())
        return wav_file

    wav_file = pcm_to_wav(audio_file)

    with open(wav_file, 'rb') as f:
        audio_bytes = f.read()
        st.audio(audio_bytes, format='audio/wav')

# Display webcam stream
FRAME_WINDOW = st.image([])

# Function to update webcam frames
def update_frames():
    while True:
        frame = st.session_state['webcam'].read()
        FRAME_WINDOW.image(frame)
        time.sleep(0.03)  # ~30 FPS

# Start updating frames in a separate thread
import threading

if 'frame_thread' not in st.session_state:
    frame_thread = threading.Thread(target=update_frames, daemon=True)
    frame_thread.start()
    st.session_state['frame_thread'] = frame_thread

# Display the last prompt and response
if 'last_prompt' in st.session_state:
    response_placeholder.markdown(f"**You:** {st.session_state['last_prompt']}")

# Assuming Assistant class stores the last response
if hasattr(st.session_state['assistant'], 'last_response'):
    response_placeholder.markdown(f"**Assistant:** {st.session_state['assistant'].last_response}")

# Handle stop condition
if st.button("Stop"):
    st.session_state['webcam'].stop()
    st.session_state['stop_listening'](wait_for_stop=False)
    st.stop()

# Clean up on app close
def on_exit():
    st.session_state['webcam'].stop()
    st.session_state['stop_listening'](wait_for_stop=False)

import atexit
atexit.register(on_exit)
