from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fabricated.captions.ass import render_ass_script
from fabricated.captions.backends import get_caption_backend
from fabricated.captions.backends.base import CaptionBackendError
from fabricated.captions.chunking import chunk_transcript
from fabricated.captions.fonts import PreparedFont, prepare_font, resolve_font_preset
from fabricated.config import CaptionConfig


@dataclass(slots=True, frozen=True)
class CaptionArtifacts:
    subtitle_output: Path
    transcript_output: Path
    fonts_dir: Path
    font: PreparedFont


def generate_caption_assets(
    audio_path: Path,
    subtitle_output: Path,
    transcript_output: Path,
    caption_config: CaptionConfig,
    temp_dir: Path,
    *,
    width: int | None = None,
    height: int | None = None,
) -> CaptionArtifacts:
    backend = get_caption_backend(caption_config)
    transcript = backend.transcribe(audio_path)
    if not transcript.segments:
        raise CaptionBackendError("Caption transcription returned no segments.")

    subtitle_output.parent.mkdir(parents=True, exist_ok=True)
    transcript_output.parent.mkdir(parents=True, exist_ok=True)

    transcript_output.write_text(
        json.dumps(transcript.to_dict(), indent=2),
        encoding="utf-8",
    )

    chunks = chunk_transcript(
        transcript,
        max_words=caption_config.max_words_per_caption,
        pause_threshold=caption_config.pause_threshold_sec,
    )
    resolved_font = resolve_font_preset(caption_config.font)
    prepared_font = prepare_font(resolved_font, temp_dir / "caption-fonts")
    ass_script = render_ass_script(
        chunks,
        prepared_font,
        caption_config.style,
        video_width=width or 1920,
        video_height=height or 1080,
    )
    subtitle_output.write_text(ass_script, encoding="utf-8")

    return CaptionArtifacts(
        subtitle_output=subtitle_output,
        transcript_output=transcript_output,
        fonts_dir=prepared_font.fonts_dir,
        font=prepared_font,
    )
