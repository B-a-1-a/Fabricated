from __future__ import annotations

from importlib import import_module

from fabricated.captions.backends.base import CaptionBackend
from fabricated.config import CaptionConfig

CAPTION_BACKEND_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "faster-whisper": ("fabricated.captions.backends.faster_whisper_backend", "FasterWhisperCaptionBackend"),
    "whisperx": ("fabricated.captions.backends.whisperx_backend", "WhisperXCaptionBackend"),
}


def resolve_caption_backend_class(name: str) -> type[CaptionBackend]:
    try:
        module_name, class_name = CAPTION_BACKEND_ENTRYPOINTS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(CAPTION_BACKEND_ENTRYPOINTS))
        raise ValueError(f"Unsupported caption backend {name!r}. Expected one of: {supported}.") from exc
    module = import_module(module_name)
    return getattr(module, class_name)


def get_caption_backend(config: CaptionConfig) -> CaptionBackend:
    backend_class = resolve_caption_backend_class(config.backend)
    return backend_class(config)
