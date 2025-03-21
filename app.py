from fastapi import FastAPI, Form, HTTPException
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from keras import backend as K

app = FastAPI()

# ✅ Debugging: Print current working directory and check model existence
print("Current Working Directory:", os.getcwd())
if os.path.exists("Models"):
    print("Files in Models Directory:", os.listdir("Models"))
else:
    print("⚠️ 'Models' directory not found!")

# ✅ Registering custom F1-score function
@register_keras_serializable()
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Convert probabilities to binary values
    tp = K.sum(K.cast(y_true * y_pred, "float32"))  # True Positives
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float32"))  # False Positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float32"))  # False Negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# ✅ Register custom LSTM layer to avoid errors
@register_keras_serializable()
class CustomLSTM(LSTM):
    pass  # Inherits everything from LSTM

# ✅ Paths for model and tokenizer
MODEL_PATH = "Models/best_model.keras"  # Relative path for portability
TOKENIZER_PATH = "tokenizer.pickle"

# ✅ Ensure model & tokenizer exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"❌ Tokenizer file '{TOKENIZER_PATH}' not found!")

# ✅ Load model with custom objects
try:
    model = load_model(MODEL_PATH, custom_objects={"LSTM": CustomLSTM, "f1_score": f1_score}, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {str(e)}")

# ✅ Load tokenizer
try:
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading tokenizer: {str(e)}")

# ✅ Define sentiment labels
sentiment_classes = ["Negative", "Neutral", "Positive"]
max_len = 50  # Should match training config

def predict_class(text: str):
    """Predict sentiment class of the input text"""
    xt = tokenizer.texts_to_sequences([text])
    xt = pad_sequences(xt, padding="post", maxlen=max_len)
    yt = model.predict(xt).argmax(axis=1)
    return sentiment_classes[yt[0]]

# ✅ FastAPI Routes
@app.get("/")
def home():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/sentiment/")
def sentimental_check(text: str = Form(...)):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    prediction = predict_class(text)
    return {"input_text": text, "predicted_sentiment": prediction}
