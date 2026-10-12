from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fabricated.config import PipelineConfig


class MediaToolError(RuntimeError):
    """Raised when ffmpeg or ffprobe fails."""


@dataclass(slots=True, frozen=True)
class MediaInfo:
    path: Path
    has_audio: bool
    has_video: bool
    duration_sec: float
    sample_rate: int | None
    channels: int | None
    width: int | None
    height: int | None


def ensure_media_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise MediaToolError(f"Missing required media tool(s): {missing_list}.")


def run_media_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise MediaToolError(stderr or f"Command failed: {' '.join(args)}")
    return completed


def probe_media(input_path: Path) -> MediaInfo:
    ensure_media_tools()
    completed = run_media_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(input_path),
        ]
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    duration_raw = payload.get("format", {}).get("duration")
    duration = float(duration_raw) if duration_raw is not None else 0.0
    sample_rate = int(audio_stream["sample_rate"]) if audio_stream and audio_stream.get("sample_rate") else None
    channels = int(audio_stream["channels"]) if audio_stream and audio_stream.get("channels") else None
    width = int(video_stream["width"]) if video_stream and video_stream.get("width") else None
    height = int(video_stream["height"]) if video_stream and video_stream.get("height") else None
    return MediaInfo(
        path=input_path,
        has_audio=audio_stream is not None,
        has_video=video_stream is not None,
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        width=width,
        height=height,
    )


def extract_audio(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_f32le",
            str(output_path),
        ]
    )
    return output_path


def build_filter_chain(config: PipelineConfig) -> str:
    polish = config.polish
    filters: list[str] = []

    highpass_hz = polish.get("highpass_hz")
    if highpass_hz:
        filters.append(f"highpass=f={highpass_hz}")

    deesser = polish.get("deesser", {})
    if deesser:
        filters.append(
            "deesser="
            f"i={deesser.get('intensity', 0.15)}:"
            f"m={deesser.get('max_deess', 0.5)}:"
            f"f={deesser.get('frequency', 0.5)}"
        )

    compressor = polish.get("compressor", {})
    if compressor:
        filters.append(
            "acompressor="
            f"threshold={compressor.get('threshold', 0.09)}:"
            f"ratio={compressor.get('ratio', 2.0)}:"
            f"attack={compressor.get('attack_ms', 15)}:"
            f"release={compressor.get('release_ms', 140)}:"
            f"makeup={compressor.get('makeup', 1.2)}:"
            f"mix={compressor.get('mix', 0.9)}"
        )

    speechnorm = polish.get("speechnorm", {})
    if speechnorm:
        filters.append(
            "speechnorm="
            f"p={speechnorm.get('peak', 0.95)}:"
            f"e={speechnorm.get('expansion', 1.5)}:"
            f"c={speechnorm.get('compression', 2.0)}:"
            f"t={speechnorm.get('threshold', 0.0)}:"
            f"r={speechnorm.get('raise', 0.001)}:"
            f"f={speechnorm.get('fall', 0.001)}"
        )

    loudnorm = polish.get("loudnorm", {})
    filters.append(
        "loudnorm="
        f"I={config.target_lufs}:"
        f"LRA={loudnorm.get('lra', 7.0)}:"
        f"TP={loudnorm.get('true_peak', -1.5)}:"
        "linear=true"
    )

    limiter = polish.get("limiter", {})
    filters.append(
        "alimiter="
        f"limit={limiter.get('limit', 0.91)}:"
        f"attack={limiter.get('attack_ms', 5)}:"
        f"release={limiter.get('release_ms', 50)}:"
        "level=true:latency=true"
    )

    return ",".join(filters)


def polish_audio(
    input_path: Path,
    output_path: Path,
    *,
    config: PipelineConfig,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = build_filter_chain(config)
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-af",
            filter_chain,
            "-ac",
            str(config.channels),
            "-ar",
            str(config.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return output_path


def remux_video_with_audio(
    input_video: Path,
    input_audio: Path,
    output_video: Path,
    *,
    audio_codec: str,
    audio_bitrate: str,
) -> Path:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(input_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_video),
        ]
    )
    return output_video


def build_subtitle_filter(
    subtitle_path: Path,
    fonts_dir: Path,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    original_size = f":original_size={width}x{height}" if width and height else ""
    return (
        "ass="
        f"filename='{_escape_filter_path(subtitle_path)}':"
        f"fontsdir='{_escape_filter_path(fonts_dir)}'"
        f"{original_size}"
    )


def burn_subtitles_into_video(
    input_video: Path,
    input_audio: Path,
    subtitle_path: Path,
    fonts_dir: Path,
    output_video: Path,
    *,
    audio_codec: str,
    audio_bitrate: str,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    subtitle_filter = build_subtitle_filter(subtitle_path, fonts_dir, width=width, height=height)
    run_media_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-i",
            str(input_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            subtitle_filter,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            audio_codec,
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_video),
        ]
    )
    return output_video


def _escape_filter_path(path: Path) -> str:
    value = str(path)
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
