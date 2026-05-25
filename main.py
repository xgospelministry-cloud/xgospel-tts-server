"""XGospel BibleTTS FastAPI server.

Endpoints:
  GET  /healthz   -> liveness + loaded model list
  POST /tts       -> { language, text } -> audio/wav stream

All 5 VITS models are loaded into memory at startup (~5-7GB RAM total),
so /tts requests have no cold-start cost.
"""
import io
import os
import unicodedata
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from TTS.api import TTS

MODELS = {
    "asante-twi": "tts_models/tw_asante/openbible/vits",
    "ewe":        "tts_models/ewe/openbible/vits",
    "yoruba":     "tts_models/yor/openbible/vits",
    "hausa":      "tts_models/hau/openbible/vits",
}

QUALITY = {
    "asante-twi": "high",
    "ewe":        "high",
    "yoruba":     "high",
    "hausa":      "medium",
}

API_KEY = os.environ.get("API_KEY")
MAX_CHARS = 500

app = FastAPI(title="XGospel BibleTTS")
loaded: dict[str, TTS] = {}


@app.on_event("startup")
def load_models() -> None:
    for lang, model_name in MODELS.items():
        print(f"Loading {lang}: {model_name}")
        loaded[lang] = TTS(model_name=model_name, progress_bar=False)
    print(f"Loaded {len(loaded)} models.")


class TTSRequest(BaseModel):
    language: str = Field(..., description="asante-twi | ewe | yoruba | igbo | hausa")
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "models": list(loaded.keys())}


@app.post("/tts")
def synthesize(req: TTSRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    lang = req.language.lower().strip()
    if lang not in loaded:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language '{lang}'. Supported: {list(loaded.keys())}",
        )

    # NFC keeps Igbo composed forms (ọ ụ ị) intact through any client mangling.
    text = unicodedata.normalize("NFC", req.text)

    tts = loaded[lang]
    sample_rate = tts.synthesizer.output_sample_rate
    waveform = tts.tts(text=text)

    audio = np.array(waveform, dtype=np.float32)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"X-Quality": QUALITY[lang]},
    )
