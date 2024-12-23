# utils/__init__.py
# to define what is accessible when the package is imported. 
# In this case, we are importing the classes WebcamStream, Assistant, and the functions audio_callback, setup_audio, and load_configuration. 
# This is done by importing these classes and functions from their respective modules and then defining them in the __init__.py file. This allows us to access these classes and functions directly from the package when importing it.
from .webcam import WebcamStream
from .assistant import Assistant
from .audio import audio_callback, setup_audio
from .config import load_configuration