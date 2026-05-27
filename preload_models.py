"""Download every TTS model the server uses so they bake into the Docker
image. Runs at `docker build` time. Without this, the container would
download multi-GB of weights on first start, blocking the first request
for minutes.

Two model families are pre-fetched:
  1. The 4 BibleTTS Coqui-distributed VITS models (Twi / Ewe / Yoruba /
     Hausa) — registered under standard `tts_models/...` identifiers and
     downloaded by Coqui's own model manager.
  2. The community `multilingual-tts/VITS-OpenBible-Igbo` model, hosted
     on Hugging Face. Facebook MMS never released an Igbo TTS model and
     BibleTTS itself didn't either — investigation in May 2026 confirmed
     `https://dl.fbaipublicfiles.com/mms/tts/{ig,ibo}.tar.gz` returns 403,
     and `facebook/mms-tts-ibo` on HF is gated/private. This community
     model is the only viable Coqui-compatible Igbo VITS in the wild;
     loaded at runtime via TTS.utils.synthesizer.Synthesizer (multispeaker).
"""
from TTS.api import TTS
from huggingface_hub import hf_hub_download

MODELS = [
    "tts_models/tw_asante/openbible/vits",
    "tts_models/ewe/openbible/vits",
    "tts_models/yor/openbible/vits",
    "tts_models/hau/openbible/vits",
]

for model_name in MODELS:
    print(f"==> Downloading {model_name}")
    TTS(model_name=model_name, progress_bar=False)
    print(f"    done.\n")

# Igbo lives on a different host and uses a different loader — pull each
# file individually so the HF cache is populated at image-build time.
IGBO_HF_REPO = "multilingual-tts/VITS-OpenBible-Igbo"
IGBO_HF_FILES = ("model_last.pth", "config.json", "speakers.pth")
print(f"==> Downloading Igbo model from Hugging Face: {IGBO_HF_REPO}")
for fname in IGBO_HF_FILES:
    print(f"    - {fname}")
    hf_hub_download(IGBO_HF_REPO, fname)
print("    done.\n")

print("All TTS models downloaded.")
