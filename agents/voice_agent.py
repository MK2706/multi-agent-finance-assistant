from gtts import gTTS
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/speak")
def speak():
    text = "Good morning. Here is your market brief."
    speech = gTTS(text)
    speech.save("voice.mp3")
    return {"status": "Voice saved as voice.mp3"}