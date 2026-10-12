from __future__ import annotations

from pathlib import Path

import soundfile as sf

from fabricated.captions.types import Transcript, TranscriptSegment, TranscriptWord
from fabricated.cli import _build_parser
from fabricated.config import PipelineConfig
from fabricated.pipeline.process_audio import enhance_audio
from fabricated.pipeline.process_video import process_video


class DummyProvider:
    name = "dummy"
    last_sample_rate: int | None = None
    last_channels: int | None = None
    last_bandwidth_extend: bool | None = None

    def __init__(self, config: PipelineConfig):
        self.config = config

    def enhance(self, input_audio: Path, output_audio: Path):
        audio, sample_rate = sf.read(input_audio, dtype="float32", always_2d=True)
        DummyProvider.last_sample_rate = sample_rate
        DummyProvider.last_channels = audio.shape[1]
        DummyProvider.last_bandwidth_extend = self.config.bandwidth_extend
        sf.write(output_audio, audio[:, 0], sample_rate, subtype="FLOAT")
        return None


class DummyCaptionBackend:
    def __init__(self, config):
        self.config = config

    def transcribe(self, audio_path: Path):
        return Transcript(
            backend=self.config.backend,
            language="en",
            model=self.config.model or "dummy-small",
            text="hello world again",
            segments=(
                TranscriptSegment(
                    text="hello world again",
                    start=0.0,
                    end=0.9,
                    words=(
                        TranscriptWord("hello", 0.0, 0.3),
                        TranscriptWord("world", 0.3, 0.6),
                        TranscriptWord("again", 0.6, 0.9),
                    ),
                ),
            ),
        )


def test_process_video_exports_audio_and_mp4(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "clearvoice",
        (__name__, "DummyProvider"),
    )
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _make_video_with_audio(source, sample_rate=48_000, channels=1)

    result = process_video(source, output, PipelineConfig.from_options(preset="studio"))

    assert result.video_output == output
    assert result.audio_output == output.with_suffix(".wav")
    assert output.exists()
    assert result.audio_output.exists()
    assert result.subtitle_output is None
    assert result.transcript_output is None


def test_process_video_normalizes_to_48k_mono(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "deepfilternet",
        (__name__, "DummyProvider"),
    )
    source = tmp_path / "stereo.mp4"
    output = tmp_path / "cleaned.mp4"
    _make_video_with_audio(source, sample_rate=44_100, channels=2)

    process_video(source, output, PipelineConfig.from_options(preset="fast"))

    assert DummyProvider.last_sample_rate == 48_000
    assert DummyProvider.last_channels == 1


def test_process_video_fails_when_audio_track_is_missing(tmp_path):
    source = tmp_path / "silent.mp4"
    _make_video_without_audio(source)
    try:
        process_video(source, tmp_path / "unused.mp4", PipelineConfig.from_options())
    except RuntimeError as exc:
        assert "No audio stream" in str(exc)
    else:
        raise AssertionError("Expected process_video() to fail when the input has no audio track.")


def test_process_video_supports_audio_only_export(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "clearvoice",
        (__name__, "DummyProvider"),
    )
    source = tmp_path / "input.mp4"
    output = tmp_path / "audio-only.wav"
    _make_video_with_audio(source, sample_rate=48_000, channels=1)

    result = process_video(
        source,
        output,
        PipelineConfig.from_options(preset="studio", remux_video=False),
    )

    assert result.video_output is None
    assert result.audio_output == output
    assert output.exists()


def test_bandwidth_extension_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "clearvoice",
        (__name__, "DummyProvider"),
    )
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _make_video_with_audio(source, sample_rate=48_000, channels=1)

    process_video(source, output, PipelineConfig.from_options(preset="studio"))

    assert DummyProvider.last_bandwidth_extend is False


def test_enhance_audio_exports_wav(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "deepfilternet",
        (__name__, "DummyProvider"),
    )
    source = tmp_path / "input.wav"
    output = tmp_path / "output.wav"
    _make_audio_fixture(source, sample_rate=48_000, channels=1)

    result = enhance_audio(source, output, PipelineConfig.from_options(preset="fast"))

    assert result.video_output is None
    assert result.audio_output == output
    assert output.exists()


def test_cli_process_out_is_optional():
    parser = _build_parser()
    args = parser.parse_args(["process", "input.mov"])
    assert args.out is None
    assert args.captions is False


def test_cli_enhance_audio_out_is_optional():
    parser = _build_parser()
    args = parser.parse_args(["enhance-audio", "input.wav"])
    assert args.out is None


def test_cli_process_accepts_caption_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "process",
            "input.mov",
            "--captions",
            "--caption-backend",
            "faster-whisper",
            "--caption-style",
            "impact",
            "--caption-font",
            "futura",
            "--caption-language",
            "en",
        ]
    )
    assert args.captions is True
    assert args.caption_backend == "faster-whisper"
    assert args.caption_style == "impact"
    assert args.caption_font == "futura"
    assert args.caption_language == "en"


def test_process_video_with_captions_outputs_ass_transcript_and_burned_video(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "clearvoice",
        (__name__, "DummyProvider"),
    )
    monkeypatch.setitem(
        __import__("fabricated.captions.backends", fromlist=["CAPTION_BACKEND_ENTRYPOINTS"]).CAPTION_BACKEND_ENTRYPOINTS,
        "faster-whisper",
        (__name__, "DummyCaptionBackend"),
    )
    source = tmp_path / "input.mp4"
    output = tmp_path / "captioned.mp4"
    _make_video_with_audio(source, sample_rate=48_000, channels=1)

    result = process_video(
        source,
        output,
        PipelineConfig.from_options(
            preset="studio",
            captions=True,
            caption_backend="faster-whisper",
            caption_style="impact",
            caption_font="impact",
        ),
    )

    assert result.video_output == output
    assert result.audio_output == output.with_suffix(".wav")
    assert result.subtitle_output == output.with_suffix(".ass")
    assert result.transcript_output == output.with_suffix(".transcript.json")
    assert output.exists()
    assert result.audio_output.exists()
    assert result.subtitle_output.exists()
    assert result.transcript_output.exists()
    subtitle_text = result.subtitle_output.read_text(encoding="utf-8")
    assert "Impact" in subtitle_text
    assert "Dialogue: 0," in subtitle_text


def test_process_video_with_caption_font_selection_changes_ass_font(monkeypatch, tmp_path):
    monkeypatch.setitem(
        __import__("fabricated.providers", fromlist=["PROVIDER_ENTRYPOINTS"]).PROVIDER_ENTRYPOINTS,
        "clearvoice",
        (__name__, "DummyProvider"),
    )
    monkeypatch.setitem(
        __import__("fabricated.captions.backends", fromlist=["CAPTION_BACKEND_ENTRYPOINTS"]).CAPTION_BACKEND_ENTRYPOINTS,
        "faster-whisper",
        (__name__, "DummyCaptionBackend"),
    )
    source = tmp_path / "input.mp4"
    output = tmp_path / "captioned.mp4"
    _make_video_with_audio(source, sample_rate=48_000, channels=1)

    result = process_video(
        source,
        output,
        PipelineConfig.from_options(
            preset="studio",
            captions=True,
            caption_backend="faster-whisper",
            caption_style="highlight",
            caption_font="futura",
        ),
    )

    subtitle_text = result.subtitle_output.read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "Futura" in subtitle_text


def _make_audio_fixture(path: Path, *, sample_rate: int, channels: int) -> None:
    import subprocess

    if channels == 1:
        source = "sine=frequency=220:sample_rate={sample_rate}:duration=1.2"
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source.format(sample_rate=sample_rate),
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    else:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=220:sample_rate={sample_rate}:duration=1.2",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={sample_rate}:duration=1.2",
            "-filter_complex",
            "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_s16le",
            str(path),
        ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _make_video_with_audio(path: Path, *, sample_rate: int, channels: int) -> None:
    import subprocess

    temp_audio = path.with_suffix(".wav")
    _make_audio_fixture(temp_audio, sample_rate=sample_rate, channels=channels)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1.2",
            "-i",
            str(temp_audio),
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    temp_audio.unlink()


def _make_video_without_audio(path: Path) -> None:
    import subprocess

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1.0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
