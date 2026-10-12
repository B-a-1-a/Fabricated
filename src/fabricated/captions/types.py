from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TranscriptWord:
    text: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TranscriptWord":
        return cls(
            text=str(payload["text"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
        )


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: tuple[TranscriptWord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TranscriptSegment":
        words = tuple(TranscriptWord.from_dict(word) for word in payload.get("words", []))
        return cls(
            text=str(payload["text"]),
            start=float(payload["start"]),
            end=float(payload["end"]),
            words=words,
        )


@dataclass(slots=True, frozen=True)
class Transcript:
    backend: str
    language: str
    model: str
    text: str
    segments: tuple[TranscriptSegment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "language": self.language,
            "model": self.model,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Transcript":
        return cls(
            backend=str(payload["backend"]),
            language=str(payload["language"]),
            model=str(payload["model"]),
            text=str(payload["text"]),
            segments=tuple(TranscriptSegment.from_dict(segment) for segment in payload.get("segments", [])),
        )

    def iter_words(self) -> tuple[TranscriptWord, ...]:
        words: list[TranscriptWord] = []
        for segment in self.segments:
            words.extend(segment.words)
        return tuple(words)


@dataclass(slots=True, frozen=True)
class CaptionChunk:
    start: float
    end: float
    text: str
    words: tuple[TranscriptWord, ...]
