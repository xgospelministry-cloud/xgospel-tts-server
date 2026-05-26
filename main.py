"""XGospel BibleTTS FastAPI server.

Endpoints:
  GET  /healthz   -> liveness + loaded model list
  POST /tts       -> { language, text } -> audio/mpeg stream

All 4 VITS models are loaded into memory at startup (~3-4GB RAM total),
so /tts requests have no cold-start cost.

Behaviour:
- Synthesise at the model's native sample rate (48 kHz for BibleTTS VITS).
- Encode to MP3 96 kbps mono via ffmpeg piped through stdin/stdout.
- Persist the MP3 to /srv/audio/v1/{lang}/{hash}.mp3 so audio.xgospel.net
  can serve subsequent identical requests as a static file. The hash is
  FNV-1a-32 over (lang, voiceVersion, text), matched bit-for-bit by the
  client so the URL can be reconstructed without a server round-trip.
- Return the MP3 bytes inline; the client caches them locally too.
"""
import io
import os
import subprocess
import unicodedata
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import Response
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
VOICE_VERSION = 1
# Bind-mounted host directory shared with the audio-nginx container so the
# files we write here are immediately served at audio.xgospel.net.
AUDIO_DIR = Path("/srv/audio")
MP3_BITRATE = "96k"

app = FastAPI(title="XGospel BibleTTS")
loaded: dict[str, TTS] = {}


@app.on_event("startup")
def load_models() -> None:
    for lang, model_name in MODELS.items():
        print(f"Loading {lang}: {model_name}")
        loaded[lang] = TTS(model_name=model_name, progress_bar=False)
    print(f"Loaded {len(loaded)} models.")
    try:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as err:  # noqa: BLE001
        # Non-fatal — the synth path falls back to in-memory return on save fail.
        print(f"WARN: could not create {AUDIO_DIR}: {err}")


class TTSRequest(BaseModel):
    language: str = Field(..., description="asante-twi | ewe | yoruba | hausa")
    text: str = Field(..., min_length=1, max_length=MAX_CHARS)


def fnv1a_js(s: str) -> str:
    """Match the client's `fnv1a` in `src/utils/aiAudio.js` exactly.

    JS iterates UTF-16 code units via String.prototype.charCodeAt. To match
    that on the Python side we encode to UTF-16 LE and read 2 bytes at a
    time. Output is base36 (matching JS `.toString(36)`) so the resulting
    hash strings are byte-identical to what the client computes.
    """
    h = 0x811C9DC5
    utf16 = s.encode("utf-16-le")
    for i in range(0, len(utf16), 2):
        code_unit = utf16[i] | (utf16[i + 1] << 8)
        h ^= code_unit
        h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) & 0xFFFFFFFF
    if h == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while h > 0:
        out = digits[h % 36] + out
        h //= 36
    return out


def cache_path(language: str, text: str) -> Path:
    """Path under AUDIO_DIR for a given (language, text). Must match the
    URL the client constructs in aiAudio.js cdnUrl()."""
    h = fnv1a_js(f"{language}:{VOICE_VERSION}:{text}")
    return AUDIO_DIR / f"v{VOICE_VERSION}" / language / f"{h}.mp3"


def wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Encode WAV bytes → MP3 96 kbps mono via ffmpeg stdio pipes.

    Faster than touching disk — ~50 ms for a 2-second utterance on this VPS.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "wav", "-i", "pipe:0",
            "-b:a", MP3_BITRATE, "-ac", "1",
            "-f", "mp3", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def save_atomic(target: Path, data: bytes) -> None:
    """Write `data` to a temp file and rename onto `target` so concurrent
    readers (nginx) never observe a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


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

    # NFC keeps composed forms intact through any client encoding mangling.
    text = unicodedata.normalize("NFC", req.text)

    target = cache_path(lang, text)
    if target.exists() and target.stat().st_size > 0:
        # Disk hit — skip synthesis entirely. Rare in practice because the
        # client checks the CDN URL first, but covers the race where two
        # clients hit /tts for the same string before the first finishes.
        return Response(
            content=target.read_bytes(),
            media_type="audio/mpeg",
            headers={"X-Quality": QUALITY[lang], "X-Cache": "disk"},
        )

    tts = loaded[lang]
    sample_rate = tts.synthesizer.output_sample_rate
    waveform = tts.tts(text=text)

    audio = np.array(waveform, dtype=np.float32)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    try:
        mp3_bytes = wav_to_mp3(wav_buf.getvalue())
    except subprocess.CalledProcessError as err:
        # Fall back to WAV if ffmpeg fails for any reason (corrupt input,
        # missing binary, etc). Client still plays it; just larger payload.
        print(f"WARN: ffmpeg failed for '{lang}': {err.stderr!r}")
        return Response(
            content=wav_buf.getvalue(),
            media_type="audio/wav",
            headers={"X-Quality": QUALITY[lang], "X-Cache": "miss-wav"},
        )

    try:
        save_atomic(target, mp3_bytes)
    except Exception as err:  # noqa: BLE001
        # Non-fatal — still return the MP3 to the caller, just skip the
        # persistent cache (next request will re-synthesise).
        print(f"WARN: could not save {target}: {err}")

    return Response(
        content=mp3_bytes,
        media_type="audio/mpeg",
        headers={"X-Quality": QUALITY[lang], "X-Cache": "miss"},
    )
