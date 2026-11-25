from fastapi import FastAPI, HTTPException # HTTPException: let us handle exceptions
import re # used for text cleaning (removing punctuation, spaces, etc.).
from pydantic import BaseModel, Field
from keras.models import load_model
import pickle
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.responses import FileResponse
import numpy as np

"""
Some constants we will add here:
1. Model Path (BILSTM)
2. Tokeniser Path
3. Max Sequence length
4. Emotion labels
5. Emotion emojis
"""

# Model Path 
model_path = "artifacts/BILSTM_model.keras"

# Tokeniser
tokeniser_path = "artifacts/tokenizer.pkl"

# max sequence length
max_sequence_length = 50

# Emotion labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Emotion emojis
emotion_emojis = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

# Preprocess the upcomign text: cleans raw text so it matches the format used while training
# A. convert the text to lowercase
# B. Remove Aposthrophes
# C. Remove special char and punctuations
# D. Remove extra spaces

def preprocess_text(text: str)-> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Request and response schemas
# 1. text input
# 2. prediciton Response
# 3. health response

class TextInput(BaseModel):
    text : str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
        )

class PredicitonResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Model loading and lifespan management: we will load the model and the server once the server starts

dl_model = {} # An empty dictionary that will hold our loaded model + tokenizer for the whole app's lifetime.

@asynccontextmanager
async def lifespan(app: FastAPI): # FastAPI runs the code before yield when the server starts, and the code after yield when the server shuts down.
    print("loading model and tokenizer...")
    dl_model["BILSTM"] = load_model(model_path)
    with open(tokeniser_path, 'rb') as file:
        dl_model["tokenizer"] = pickle.load(file)
    print("model loaded succesfully...")

    yield # Server is ready, start handling requests." Everything above runs once before any request comes in.

    dl_model.clear()

# Mount the static file to the FastApi app
# 1. Enable CORS

app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory= "static"), name= "static")

# Api Endpoints
# 1. Server UI at homepage ("/")
# 2. Health check endpoint ("/health")
# 3 Predict emotion endpoint ("/predict")

@app.get("/", include_in_schema= False)
def server_ui():
    return FileResponse('static/index.html')

@app.get("/health", response_model = HealthResponse)
def health_check():
    return HealthResponse(status= "Server is running", model_loaded = bool(dl_model))

@app.post("/predict", response_model = PredicitonResponse)
def predict(text_input: TextInput):
    
    # 1. cleans the input sentences
    LsTM_model = dl_model.get("BILSTM")
    tokenizer_model = dl_model.get("tokenizer") 

    if LsTM_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail= "Model is not loaded yet. Please try again")
    # 5. return prediction emotion

    #1. Cleans the input sentences.
    cleaned_text = preprocess_text(text_input.text)

    # 2. convert the words into numeric using tokenizer & # 3. pad the sequence
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen = max_sequence_length,
        padding = "post",
        truncating = "post"
    )

    # 4. run prediction
    probabilites = LsTM_model.predict(padded_sequence)[0]
    top_emotion_index = int(np.argmax(probabilites))
    all_probabilites = {label: float(prob) for prob, label in zip(probabilites, emotion_labels)}

    return PredicitonResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilites[top_emotion_index]), 
        all_probabilites = all_probabilites
    )