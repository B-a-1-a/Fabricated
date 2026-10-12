from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fabricated.config import PipelineConfig, ProcessResult
from fabricated.media.ffmpeg import MediaToolError, extract_audio, polish_audio, probe_media
from fabricated.providers import get_provider


def enhance_audio(input_path: Path | str, output_path: Path | str, config: PipelineConfig) -> ProcessResult:
    source = Path(input_path)
    target_audio = Path(output_path)
    if target_audio.suffix.lower() != ".wav":
        raise ValueError("enhance-audio requires a .wav output path.")

    info = probe_media(source)
    if not info.has_audio:
        raise MediaToolError(f"No audio stream found in {source}.")

    temp_dir = Path(tempfile.mkdtemp(prefix="fabricated-"))
    try:
        prepared_audio = extract_audio(
            source,
            temp_dir / "prepared.wav",
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        provider = get_provider(config)
        enhanced_audio = temp_dir / "enhanced.wav"
        provider.enhance(prepared_audio, enhanced_audio)
        polished_audio = polish_audio(enhanced_audio, target_audio, config=config)
        polished_info = probe_media(polished_audio)
        result_temp_dir = temp_dir if config.keep_temp else None
        return ProcessResult(
            video_output=None,
            audio_output=polished_audio,
            provider=config.provider,
            sample_rate=polished_info.sample_rate or config.sample_rate,
            duration_sec=polished_info.duration_sec or info.duration_sec,
            temp_dir=result_temp_dir,
            subtitle_output=None,
            transcript_output=None,
        )
    finally:
        if not config.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
