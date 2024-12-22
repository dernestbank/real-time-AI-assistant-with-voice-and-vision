import base64
from threading import Lock, Thread

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()


class WebcamStream:
    def __init__(self):
        """
        Initialize the WebcamStream.

        This sets up the stream, reads the first frame, and sets the running flag
        to False and the lock to an empty Lock.
        """
        self.stream = VideoCapture(index=0)#0 is the default webcam index
        _, self.frame = self.stream.read()#read the first frame
        self.running = False
        self.lock = Lock()#create a lock to avoid race conditions

    def start(self):
        """
        Starts the thread to read frames from the webcam.

        If the thread is already running, this is a no-op. Otherwise, it sets the
        running flag to True, creates the thread, and starts it.

        Returns:
            The current instance.
        """
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """
        Update the frame read from the webcam.

        This is a blocking, infinite loop that reads frames from the webcam and
        updates the internal frame. This is designed to be run in a separate
        thread.

        The self.running flag must be set to True before calling this.

        Returns:
            None
        """
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        """
        Read the latest frame from the webcam.

        Args:
            encode: Whether or not to encode the frame as a b64 string.

        Returns:
            The latest frame from the webcam, either as a numpy array or as a
            b64 string.
        """
        self.lock.acquire()#acquire the lock to avoid race conditions
        frame = self.frame.copy()#copy the frame to avoid race conditions
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
                
        """
        Initialize the Assistant.

        Args:
            model: The model to be used for creating the inference chain.
        """
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        """
        Answer a question from the user.

        Args:
            prompt: The text from the user that triggered this call.
            image: The latest image captured from the webcam, encoded as a b64
                string.

        Returns:
            None
        """
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        """
        Convert a text response to speech using the openai tts-1 model, and
        play it using the PyAudio library.

        Args:
            response (str): The text response to convert to speech.

        Returns:
            None
        """
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        """
        Create an inference chain using a system prompt, a prompt template, and a model.

        This function sets up a chat-based inference chain by configuring a system
        prompt that instructs the assistant's behavior and personality. It utilizes
        a chat prompt template that includes both text and image inputs, and pipes
        the template output through a given model. The final output is parsed as a
        string.

        The inference chain maintains chat history using a `RunnableWithMessageHistory`
        instance, which keeps track of previous interactions to provide context for
        future queries.

        Args:
            model: The language model to be used in the inference chain.

        Returns:
            RunnableWithMessageHistory: An object that supports executing the
            inference chain with chat history.
        """
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


webcam_stream = WebcamStream().start()

# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)


def audio_callback(recognizer, audio):
    """
    Called when the microphone has audio data.

    Uses the Whisper model to recognize spoken language and then asks the
    assistant to respond with the recognized text and the current webcam
    image.

    If the Whisper model raises an UnknownValueError, print an error message.
    """
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, webcam_stream.read(encode=True))

    except UnknownValueError:
        print("There was an error processing the audio.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source) # adjust for ambient noise

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    cv2.imshow("webcam", webcam_stream.read()) # display the webcam image
    if cv2.waitKey(1) in [27, ord("q")]:#27 is the ASCII code for the ESC key
        break

webcam_stream.stop()# stop the webcam stream
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
