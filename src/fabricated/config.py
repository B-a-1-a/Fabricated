from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml

ProviderName = Literal["clearvoice", "deepfilternet"]
DeviceName = Literal["auto", "mps", "cpu"]
PresetName = Literal["studio", "fast"]
CaptionBackendName = Literal["faster-whisper", "whisperx"]
CaptionStyleName = Literal["highlight", "zoom", "impact"]
CaptionFontName = Literal["impact", "avenir-next-condensed", "futura", "helvetica-neue"]

VALID_PROVIDERS = {"clearvoice", "deepfilternet"}
VALID_DEVICES = {"auto", "mps", "cpu"}
VALID_PRESETS = {"studio", "fast"}
VALID_CAPTION_BACKENDS = {"faster-whisper", "whisperx"}
VALID_CAPTION_STYLES = {"highlight", "zoom", "impact"}
VALID_CAPTION_FONTS = {"impact", "avenir-next-condensed", "futura", "helvetica-neue"}


@dataclass(slots=True, frozen=True)
class CaptionConfig:
    enabled: bool = False
    backend: CaptionBackendName = "faster-whisper"
    style: CaptionStyleName = "impact"
    font: CaptionFontName = "impact"
    language: str = "auto"
    model: str | None = None
    max_words_per_caption: int = 4
    pause_threshold_sec: float = 0.35


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    provider: ProviderName
    preset: PresetName
    device: DeviceName = "auto"
    bandwidth_extend: bool = False
    export_audio: bool = True
    remux_video: bool = True
    keep_temp: bool = False
    target_lufs: float = -16.0
    sample_rate: int = 48_000
    channels: int = 1
    provider_model: str | None = None
    super_resolution_model: str | None = None
    post_filter: bool = False
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    polish: dict[str, Any] = field(default_factory=dict)
    captions: CaptionConfig | None = None

    @classmethod
    def from_options(
        cls,
        *,
        preset: str = "studio",
        provider: str | None = None,
        device: str = "auto",
        bandwidth_extend: bool = False,
        export_audio: bool | None = None,
        remux_video: bool = True,
        keep_temp: bool = False,
        captions: bool = False,
        caption_backend: str = "faster-whisper",
        caption_style: str = "impact",
        caption_font: str = "impact",
        caption_language: str = "auto",
        caption_model: str | None = None,
    ) -> "PipelineConfig":
        preset_name = validate_preset_name(preset)
        device_name = validate_device_name(device)
        preset_data = load_preset(preset_name)
        resolved_provider = validate_provider_name(provider or preset_data["provider"])
        caption_config = None
        if captions:
            caption_config = CaptionConfig(
                enabled=True,
                backend=validate_caption_backend_name(caption_backend),
                style=validate_caption_style_name(caption_style),
                font=validate_caption_font_name(caption_font),
                language=caption_language,
                model=caption_model,
            )

        return cls(
            provider=resolved_provider,
            preset=preset_name,
            device=device_name,
            bandwidth_extend=bandwidth_extend,
            export_audio=preset_data.get("export_audio", True) if export_audio is None else export_audio,
            remux_video=remux_video,
            keep_temp=keep_temp,
            target_lufs=float(preset_data.get("target_lufs", -16.0)),
            sample_rate=int(preset_data.get("sample_rate", 48_000)),
            channels=int(preset_data.get("channels", 1)),
            provider_model=preset_data.get("provider_model"),
            super_resolution_model=preset_data.get("super_resolution_model"),
            post_filter=bool(preset_data.get("post_filter", False)),
            audio_codec=str(preset_data.get("audio_codec", "aac")),
            audio_bitrate=str(preset_data.get("audio_bitrate", "192k")),
            polish=dict(preset_data.get("polish", {})),
            captions=caption_config,
        )


@dataclass(slots=True, frozen=True)
class ProcessResult:
    video_output: Path | None
    audio_output: Path
    provider: ProviderName
    sample_rate: int
    duration_sec: float
    temp_dir: Path | None
    subtitle_output: Path | None = None
    transcript_output: Path | None = None


def load_preset(name: str) -> dict[str, Any]:
    preset_name = validate_preset_name(name)
    preset_path = resources.files("fabricated").joinpath("presets", f"{preset_name}.yaml")
    with preset_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset {preset_name!r} must deserialize to a mapping.")
    return data


def validate_provider_name(name: str) -> ProviderName:
    if name not in VALID_PROVIDERS:
        joined = ", ".join(sorted(VALID_PROVIDERS))
        raise ValueError(f"Unsupported provider {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]


def validate_preset_name(name: str) -> PresetName:
    if name not in VALID_PRESETS:
        joined = ", ".join(sorted(VALID_PRESETS))
        raise ValueError(f"Unsupported preset {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]


def validate_device_name(name: str) -> DeviceName:
    if name not in VALID_DEVICES:
        joined = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"Unsupported device {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]


def validate_caption_backend_name(name: str) -> CaptionBackendName:
    if name not in VALID_CAPTION_BACKENDS:
        joined = ", ".join(sorted(VALID_CAPTION_BACKENDS))
        raise ValueError(f"Unsupported caption backend {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]


def validate_caption_style_name(name: str) -> CaptionStyleName:
    if name not in VALID_CAPTION_STYLES:
        joined = ", ".join(sorted(VALID_CAPTION_STYLES))
        raise ValueError(f"Unsupported caption style {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]


def validate_caption_font_name(name: str) -> CaptionFontName:
    if name not in VALID_CAPTION_FONTS:
        joined = ", ".join(sorted(VALID_CAPTION_FONTS))
        raise ValueError(f"Unsupported caption font {name!r}. Expected one of: {joined}.")
    return name  # type: ignore[return-value]
