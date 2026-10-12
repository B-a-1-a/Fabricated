from __future__ import annotations

from pathlib import Path

from fabricated.captions.backends.base import CaptionBackend, CaptionBackendError
from fabricated.captions.chunking import compose_caption_text
from fabricated.captions.types import Transcript, TranscriptSegment, TranscriptWord


class FasterWhisperCaptionBackend(CaptionBackend):
    name = "faster-whisper"

    def transcribe(self, audio_path: Path) -> Transcript:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise CaptionBackendError(
                "faster-whisper is not installed in the active environment. Run `uv sync --python 3.11`."
            ) from exc

        language = None if self.config.language == "auto" else self.config.language
        model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        normalized_segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        for segment in list(segments):
            words: list[TranscriptWord] = []
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                cleaned = str(word.word).strip()
                if not cleaned:
                    continue
                words.append(
                    TranscriptWord(
                        text=cleaned,
                        start=float(word.start),
                        end=max(float(word.end), float(word.start) + 0.02),
                        confidence=float(word.probability) if getattr(word, "probability", None) is not None else None,
                    )
                )

            segment_text = str(segment.text).strip()
            if not words and segment_text:
                words = [
                    TranscriptWord(
                        text=segment_text,
                        start=float(segment.start),
                        end=max(float(segment.end), float(segment.start) + 0.02),
                        confidence=None,
                    )
                ]
            if not words:
                continue

            normalized_text = segment_text or compose_caption_text(words)
            full_text_parts.append(normalized_text)
            normalized_segments.append(
                TranscriptSegment(
                    text=normalized_text,
                    start=float(segment.start),
                    end=max(float(segment.end), float(segment.start) + 0.02),
                    words=tuple(words),
                )
            )

        return Transcript(
            backend=self.name,
            language=str(getattr(info, "language", language or "unknown")),
            model=self.model_name,
            text=" ".join(full_text_parts).strip(),
            segments=tuple(normalized_segments),
        )
