from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io
import soundfile as sf

app = FastAPI()

API_KEY = "sk_test_123456789"  
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


def analyze_audio(audio_bytes):
    # Load audio
    data, sr = sf.read(io.BytesIO(audio_bytes))
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    pitch = librosa.yin(data, fmin=50, fmax=300)
    energy = librosa.feature.rms(y=data)
    mfcc_var = np.var(mfcc)
    pitch_var = np.var(pitch)
    energy_var = np.var(energy)

    ai_score = (
        (1 / (mfcc_var + 1e-6)) +
        (1 / (pitch_var + 1e-6)) +
        (1 / (energy_var + 1e-6))
    ) / 3

    return ai_score


@app.post("/api/voice-detection")
def detect_voice(req: VoiceRequest, x_api_key: str = Header(None)):
    # API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Validation
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 supported")

    try:
        audio_bytes = base64.b64decode(req.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    ai_score = analyze_audio(audio_bytes)

    if ai_score > 0.6:
        classification = "AI_GENERATED"
        explanation = "Unnaturally consistent pitch and spectral patterns detected"
        confidence = min(ai_score, 1.0)
    else:
        classification = "HUMAN"
        explanation = "Natural pitch variation and energy fluctuations detected"
        confidence = 1 - ai_score

    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": round(float(confidence), 2),
        "explanation": explanation
    }
@app.get("/api/voice-detection/languages")
def get_supported_languages(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return {
        "status": "success",
        "supportedLanguages": SUPPORTED_LANGUAGES
    }
