from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from fabricated.captions.types import Transcript
from fabricated.config import CaptionConfig

DEFAULT_CAPTION_MODELS = {
    "faster-whisper": "small",
    "whisperx": "small",
}


class CaptionBackendError(RuntimeError):
    """Raised when caption transcription cannot proceed."""


class CaptionBackend(ABC):
    name: str

    def __init__(self, config: CaptionConfig):
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model or DEFAULT_CAPTION_MODELS[self.name]

    @abstractmethod
    def transcribe(self, audio_path: Path) -> Transcript:
        """Transcribe the given audio and return normalized word timings."""
