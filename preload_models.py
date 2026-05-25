"""Download all 5 BibleTTS VITS models so they bake into the Docker image.

Runs at `docker build` time. Without this, the container would download
~2-3GB of weights on first start, blocking the first request for minutes.
"""
from TTS.api import TTS

MODELS = [
    "tts_models/tw_asante/openbible/vits",
    "tts_models/ewe/openbible/vits",
    "tts_models/yor/openbible/vits",
    "tts_models/ig/fairseq/vits",
    "tts_models/hau/openbible/vits",
]

for model_name in MODELS:
    print(f"==> Downloading {model_name}")
    TTS(model_name=model_name, progress_bar=False)
    print(f"    done.\n")

print("All BibleTTS models downloaded.")
