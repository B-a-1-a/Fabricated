from __future__ import annotations

from pathlib import Path

from fabricated.cli import _build_parser
from fabricated.media.youtube import download_youtube_media


class _FakeDownloadError(Exception):
    pass


class _FakeYoutubeDL:
    options: dict[str, object] | None = None
    last_url: str | None = None
    last_download: bool | None = None

    def __init__(self, options: dict[str, object]):
        type(self).options = options

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, str]:
        type(self).last_url = url
        type(self).last_download = download
        outtmpl = str(type(self).options["outtmpl"])
        postprocessors = type(self).options.get("postprocessors", [])
        extension = "mp4"
        if postprocessors and postprocessors[0]["key"] == "FFmpegExtractAudio":
            extension = str(postprocessors[0]["preferredcodec"])
            if extension == "best":
                extension = "webm"
        filepath = (
            outtmpl.replace("%(title)s", "Demo Video")
            .replace("%(id)s", "abc123")
            .replace("%(ext)s", extension)
        )
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_text("media", encoding="utf-8")
        return {
            "title": "Demo Video",
            "id": "abc123",
            "ext": extension,
            "filepath": filepath,
        }

    def prepare_filename(self, info: dict[str, str]) -> str:
        return info["filepath"]


def test_cli_download_youtube_accepts_audio_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "download-youtube",
            "https://www.youtube.com/watch?v=abc123",
            "--extract-audio",
            "--audio-format",
            "wav",
        ]
    )
    assert args.out is None
    assert args.extract_audio is True
    assert args.audio_format == "wav"


def test_download_youtube_uses_explicit_mp4_output(monkeypatch, tmp_path):
    monkeypatch.setattr("fabricated.media.youtube.ensure_media_tools", lambda: None)
    monkeypatch.setattr(
        "fabricated.media.youtube._load_yt_dlp",
        lambda: (_FakeYoutubeDL, _FakeDownloadError),
    )
    target = tmp_path / "downloads" / "clip.mp4"

    result = download_youtube_media("https://www.youtube.com/watch?v=abc123", target)

    assert result == target
    assert target.exists()
    assert _FakeYoutubeDL.last_url == "https://www.youtube.com/watch?v=abc123"
    assert _FakeYoutubeDL.last_download is True
    assert _FakeYoutubeDL.options is not None
    assert _FakeYoutubeDL.options["outtmpl"] == str(target.with_suffix(".%(ext)s"))
    assert _FakeYoutubeDL.options["postprocessors"] == [
        {
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }
    ]


def test_download_youtube_uses_title_template_for_output_directory(monkeypatch, tmp_path):
    monkeypatch.setattr("fabricated.media.youtube.ensure_media_tools", lambda: None)
    monkeypatch.setattr(
        "fabricated.media.youtube._load_yt_dlp",
        lambda: (_FakeYoutubeDL, _FakeDownloadError),
    )

    result = download_youtube_media("https://youtu.be/abc123", tmp_path)

    assert result == tmp_path / "Demo Video [abc123].mp4"
    assert result.exists()
    assert _FakeYoutubeDL.options is not None
    assert _FakeYoutubeDL.options["outtmpl"] == str(tmp_path / "%(title)s [%(id)s].%(ext)s")


def test_download_youtube_can_extract_audio_to_explicit_output(monkeypatch, tmp_path):
    monkeypatch.setattr("fabricated.media.youtube.ensure_media_tools", lambda: None)
    monkeypatch.setattr(
        "fabricated.media.youtube._load_yt_dlp",
        lambda: (_FakeYoutubeDL, _FakeDownloadError),
    )
    target = tmp_path / "downloads" / "clip.wav"

    result = download_youtube_media(
        "https://www.youtube.com/watch?v=abc123",
        target,
        extract_audio=True,
        audio_format="wav",
    )

    assert result == target
    assert target.exists()
    assert _FakeYoutubeDL.options is not None
    assert _FakeYoutubeDL.options["format"] == "bestaudio/best"
    assert _FakeYoutubeDL.options["postprocessors"] == [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }
    ]


def test_download_youtube_can_extract_audio_with_dynamic_extension(monkeypatch, tmp_path):
    monkeypatch.setattr("fabricated.media.youtube.ensure_media_tools", lambda: None)
    monkeypatch.setattr(
        "fabricated.media.youtube._load_yt_dlp",
        lambda: (_FakeYoutubeDL, _FakeDownloadError),
    )
    target = tmp_path / "downloads" / "clip.m4a"

    result = download_youtube_media(
        "https://www.youtube.com/watch?v=abc123",
        target,
        extract_audio=True,
        audio_format="best",
    )

    assert result == tmp_path / "downloads" / "clip.webm"
    assert result.exists()
    assert _FakeYoutubeDL.options is not None
    assert _FakeYoutubeDL.options["outtmpl"] == str(tmp_path / "downloads" / "clip.%(ext)s")
