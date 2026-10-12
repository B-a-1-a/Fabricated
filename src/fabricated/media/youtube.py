from __future__ import annotations

from pathlib import Path
from typing import Any

from fabricated.media.ffmpeg import ensure_media_tools

DEFAULT_MEDIA_TEMPLATE = "%(title)s [%(id)s].%(ext)s"
SUPPORTED_AUDIO_FORMATS = ("best", "mp3", "m4a", "wav", "flac", "opus")
_AUDIO_EXTENSIONS = {
    "best": None,
    "mp3": "mp3",
    "m4a": "m4a",
    "wav": "wav",
    "flac": "flac",
    "opus": "opus",
}


class YouTubeDownloadError(RuntimeError):
    """Raised when yt-dlp cannot download a video or audio track."""


def download_youtube_media(
    url: str,
    output_path: Path | str | None = None,
    *,
    extract_audio: bool = False,
    audio_format: str = "mp3",
) -> Path:
    ensure_media_tools()
    youtube_dl_cls, download_error_cls = _load_yt_dlp()
    outtmpl, explicit_output = _resolve_output_template(
        output_path,
        extract_audio=extract_audio,
        audio_format=audio_format,
    )

    try:
        with youtube_dl_cls(_build_options(outtmpl, extract_audio=extract_audio, audio_format=audio_format)) as downloader:
            info = downloader.extract_info(url, download=True)
            media_info = _coerce_media_info(info)
            if explicit_output is not None:
                return explicit_output
            return _resolve_downloaded_path(downloader, media_info)
    except download_error_cls as exc:
        raise YouTubeDownloadError(str(exc)) from exc


def download_youtube_video(url: str, output_path: Path | str | None = None) -> Path:
    return download_youtube_media(url, output_path)


def _build_options(outtmpl: str, *, extract_audio: bool, audio_format: str) -> dict[str, Any]:
    if extract_audio:
        if audio_format not in SUPPORTED_AUDIO_FORMATS:
            supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
            raise ValueError(f"Unsupported audio format {audio_format!r}. Expected one of: {supported}.")
        return {
            "format": "bestaudio/best",
            "noplaylist": True,
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": audio_format,
                }
            ],
        }

    return {
        "format": "bestvideo*[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "outtmpl": outtmpl,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }


def _load_yt_dlp() -> tuple[type[Any], type[Exception]]:
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:  # pragma: no cover - exercised only in broken environments
        raise YouTubeDownloadError(
            "yt-dlp is not installed. Run `uv sync --python 3.11` to install downloader dependencies."
        ) from exc
    return YoutubeDL, DownloadError


def _resolve_output_template(
    output_path: Path | str | None,
    *,
    extract_audio: bool,
    audio_format: str,
) -> tuple[str, Path | None]:
    if output_path is None:
        return str(Path.cwd() / DEFAULT_MEDIA_TEMPLATE), None

    candidate = Path(output_path)
    if candidate.exists() and candidate.is_dir():
        candidate.mkdir(parents=True, exist_ok=True)
        return str(candidate / DEFAULT_MEDIA_TEMPLATE), None
    if candidate.suffix == "":
        candidate.mkdir(parents=True, exist_ok=True)
        return str(candidate / DEFAULT_MEDIA_TEMPLATE), None

    target_extension = _target_extension(extract_audio=extract_audio, audio_format=audio_format)
    target = candidate.with_suffix(f".{target_extension}") if target_extension is not None else candidate.with_suffix(".%(ext)s")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_extension is None:
        return str(target), None
    return str(target.with_suffix(".%(ext)s")), target


def _target_extension(*, extract_audio: bool, audio_format: str) -> str | None:
    if not extract_audio:
        return "mp4"
    return _AUDIO_EXTENSIONS.get(audio_format)


def _coerce_media_info(info: Any) -> dict[str, Any]:
    if isinstance(info, dict) and info.get("_type") == "playlist":
        entries = info.get("entries") or []
        info = next((entry for entry in entries if entry), None)
    if not isinstance(info, dict):
        raise YouTubeDownloadError("yt-dlp did not return media metadata.")
    return info


def _resolve_downloaded_path(downloader: Any, info: dict[str, Any]) -> Path:
    filepath = info.get("filepath")
    if filepath:
        return Path(filepath)
    return Path(downloader.prepare_filename(info))
