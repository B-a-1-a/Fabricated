from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fabricated.captions.service import generate_caption_assets
from fabricated.config import PipelineConfig, ProcessResult
from fabricated.media.ffmpeg import (
    MediaToolError,
    burn_subtitles_into_video,
    extract_audio,
    polish_audio,
    probe_media,
    remux_video_with_audio,
)
from fabricated.providers import get_provider


def process_video(input_path: Path | str, output_path: Path | str | None, config: PipelineConfig) -> ProcessResult:
    source = Path(input_path)
    media_info = probe_media(source)
    if not media_info.has_video:
        raise MediaToolError(f"No video stream found in {source}.")
    if not media_info.has_audio:
        raise MediaToolError(f"No audio stream found in {source}.")

    final_video_output = _resolve_video_output(source, output_path, config)
    final_audio_output = _resolve_audio_output(source, output_path, config)
    final_subtitle_output = _resolve_subtitle_output(source, output_path)
    final_transcript_output = _resolve_transcript_output(source, output_path)

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
        polished_audio = polish_audio(enhanced_audio, final_audio_output, config=config)
        subtitle_output: Path | None = None
        transcript_output: Path | None = None
        fonts_dir: Path | None = None

        if config.captions and config.captions.enabled:
            caption_artifacts = generate_caption_assets(
                polished_audio,
                final_subtitle_output,
                final_transcript_output,
                config.captions,
                temp_dir,
                width=media_info.width,
                height=media_info.height,
            )
            subtitle_output = caption_artifacts.subtitle_output
            transcript_output = caption_artifacts.transcript_output
            fonts_dir = caption_artifacts.fonts_dir

        video_output: Path | None = None
        if config.remux_video:
            assert final_video_output is not None
            if subtitle_output is not None and fonts_dir is not None:
                video_output = burn_subtitles_into_video(
                    source,
                    polished_audio,
                    subtitle_output,
                    fonts_dir,
                    final_video_output,
                    audio_codec=config.audio_codec,
                    audio_bitrate=config.audio_bitrate,
                    width=media_info.width,
                    height=media_info.height,
                )
            else:
                video_output = remux_video_with_audio(
                    source,
                    polished_audio,
                    final_video_output,
                    audio_codec=config.audio_codec,
                    audio_bitrate=config.audio_bitrate,
                )

        polished_info = probe_media(polished_audio)
        result_temp_dir = temp_dir if config.keep_temp else None
        return ProcessResult(
            video_output=video_output,
            audio_output=polished_audio,
            provider=config.provider,
            sample_rate=polished_info.sample_rate or config.sample_rate,
            duration_sec=polished_info.duration_sec or media_info.duration_sec,
            temp_dir=result_temp_dir,
            subtitle_output=subtitle_output,
            transcript_output=transcript_output,
        )
    finally:
        if not config.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_video_output(source: Path, output_path: Path | str | None, config: PipelineConfig) -> Path | None:
    if not config.remux_video:
        return None
    if output_path is None:
        return source.with_name(f"{source.stem}.cleaned.mp4")
    candidate = Path(output_path)
    if candidate.suffix.lower() != ".mp4":
        return candidate.with_suffix(".mp4")
    return candidate


def _resolve_audio_output(source: Path, output_path: Path | str | None, config: PipelineConfig) -> Path:
    if not config.export_audio:
        raise ValueError("process_video currently requires export_audio=True.")
    if output_path is None:
        return source.with_name(f"{source.stem}.cleaned.wav")
    candidate = Path(output_path)
    if not config.remux_video:
        return candidate if candidate.suffix.lower() == ".wav" else candidate.with_suffix(".wav")
    return candidate.with_suffix(".wav")


def _resolve_subtitle_output(source: Path, output_path: Path | str | None) -> Path:
    if output_path is None:
        return source.with_name(f"{source.stem}.cleaned.ass")
    return Path(output_path).with_suffix(".ass")


def _resolve_transcript_output(source: Path, output_path: Path | str | None) -> Path:
    if output_path is None:
        return source.with_name(f"{source.stem}.cleaned.transcript.json")
    candidate = Path(output_path)
    return candidate.with_suffix(".transcript.json")
