from __future__ import annotations

from pathlib import Path

from fabricated.captions.ass import render_ass_script
from fabricated.captions.backends import resolve_caption_backend_class
from fabricated.captions.backends.base import CaptionBackendError
from fabricated.captions.backends.whisperx_backend import WhisperXCaptionBackend
from fabricated.captions.chunking import chunk_transcript
from fabricated.captions.fonts import FONT_PRESETS, prepare_font, resolve_font_preset
from fabricated.captions.types import CaptionChunk, Transcript, TranscriptSegment, TranscriptWord
from fabricated.config import CaptionConfig
from fabricated.media.ffmpeg import build_subtitle_filter


def test_chunk_transcript_splits_on_punctuation_and_pause():
    transcript = Transcript(
        backend="dummy",
        language="en",
        model="dummy",
        text="hello world! long pause here",
        segments=(
            TranscriptSegment(
                text="hello world!",
                start=0.0,
                end=0.6,
                words=(
                    TranscriptWord("hello", 0.0, 0.2),
                    TranscriptWord("world!", 0.2, 0.6),
                ),
            ),
            TranscriptSegment(
                text="long pause here",
                start=1.2,
                end=1.8,
                words=(
                    TranscriptWord("long", 1.2, 1.4),
                    TranscriptWord("pause", 1.4, 1.6),
                    TranscriptWord("here", 1.6, 1.8),
                ),
            ),
        ),
    )

    chunks = chunk_transcript(transcript, max_words=4, pause_threshold=0.35)

    assert len(chunks) == 2
    assert chunks[0].text == "hello world!"
    assert chunks[1].text == "long pause here"


def test_render_ass_script_generates_multiple_dialogues_for_active_words(tmp_path):
    font = prepare_font(resolve_font_preset("impact"), tmp_path / "fonts")
    chunk = CaptionChunk(
        start=0.0,
        end=0.9,
        text="one two three",
        words=(
            TranscriptWord("one", 0.0, 0.3),
            TranscriptWord("two", 0.3, 0.6),
            TranscriptWord("three", 0.6, 0.9),
        ),
    )

    script = render_ass_script([chunk], font, "impact", video_width=1280, video_height=720)

    assert "[Events]" in script
    assert script.count("Dialogue: 0,") == 3
    assert "WrapStyle: 0" in script
    assert "\\fs" in script
    assert "\\1c" in script


def test_render_ass_script_uses_portrait_safe_layout(tmp_path):
    font = prepare_font(resolve_font_preset("impact"), tmp_path / "fonts")
    chunk = CaptionChunk(
        start=0.0,
        end=0.6,
        text="this still fits",
        words=(
            TranscriptWord("this", 0.0, 0.2),
            TranscriptWord("still", 0.2, 0.4),
            TranscriptWord("fits", 0.4, 0.6),
        ),
    )

    script = render_ass_script([chunk], font, "impact", video_width=1080, video_height=1920)

    assert "PlayResX: 1080" in script
    assert "PlayResY: 1920" in script
    assert "WrapStyle: 0" in script
    assert "Style: Default,Impact,75," in script
    assert ",2,86,86,268,1" in script


def test_font_resolution_falls_back_to_helvetica(monkeypatch, tmp_path):
    fallback_path = tmp_path / "fallback.ttf"
    fallback_path.write_bytes(b"font")
    monkeypatch.setitem(
        FONT_PRESETS,
        "impact",
        {"family": "Impact", "paths": [tmp_path / "missing-impact.ttf"]},
    )
    monkeypatch.setitem(
        FONT_PRESETS,
        "helvetica-neue",
        {"family": "Helvetica Neue", "paths": [fallback_path]},
    )

    resolved = resolve_font_preset("impact")

    assert resolved.family_name == "Helvetica Neue"
    assert resolved.fallback_used is True


def test_build_subtitle_filter_includes_ass_and_fontsdir(tmp_path):
    subtitle_path = tmp_path / "captions.ass"
    fonts_dir = tmp_path / "fonts"
    filter_value = build_subtitle_filter(subtitle_path, fonts_dir, width=1920, height=1080)
    assert "ass=" in filter_value
    assert "fontsdir=" in filter_value
    assert "original_size=1920x1080" in filter_value


def test_caption_backend_rejects_unknown_name():
    try:
        resolve_caption_backend_class("bogus")
    except ValueError as exc:
        assert "Unsupported caption backend" in str(exc)
    else:
        raise AssertionError("Expected caption backend resolution to fail.")


def test_whisperx_backend_requires_uv_in_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    backend = WhisperXCaptionBackend(CaptionConfig(enabled=True, backend="whisperx"))
    try:
        backend.transcribe(Path("/tmp/input.wav"))
    except CaptionBackendError as exc:
        assert "requires `uv`" in str(exc)
    else:
        raise AssertionError("Expected whisperx backend to fail without uv.")
