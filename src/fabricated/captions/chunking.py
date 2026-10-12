from __future__ import annotations

from fabricated.captions.types import CaptionChunk, Transcript, TranscriptWord

SENTENCE_BREAK_CHARS = tuple(".!?;:")
NO_SPACE_PREFIXES = tuple(".,!?;:)]}%")


def chunk_transcript(
    transcript: Transcript,
    *,
    max_words: int = 4,
    pause_threshold: float = 0.35,
) -> list[CaptionChunk]:
    flattened_words = [word for word in transcript.iter_words() if word.text and word.end > word.start]
    chunks: list[CaptionChunk] = []
    current: list[TranscriptWord] = []

    for index, word in enumerate(flattened_words):
        current.append(word)
        next_word = flattened_words[index + 1] if index + 1 < len(flattened_words) else None
        should_break = (
            len(current) >= max_words
            or _ends_sentence(word.text)
            or (
                next_word is not None
                and max(0.0, next_word.start - word.end) >= pause_threshold
            )
        )
        if should_break:
            chunks.append(_build_chunk(current))
            current = []

    if current:
        chunks.append(_build_chunk(current))

    return chunks


def compose_caption_text(words: list[TranscriptWord] | tuple[TranscriptWord, ...]) -> str:
    text_parts: list[str] = []
    for word in words:
        token = word.text.strip()
        if not token:
            continue
        if not text_parts:
            text_parts.append(token)
            continue
        if token[0] in NO_SPACE_PREFIXES or token.startswith("'"):
            text_parts[-1] = f"{text_parts[-1]}{token}"
        else:
            text_parts.append(token)
    return " ".join(text_parts)


def _build_chunk(words: list[TranscriptWord]) -> CaptionChunk:
    return CaptionChunk(
        start=words[0].start,
        end=words[-1].end,
        text=compose_caption_text(words),
        words=tuple(words),
    )


def _ends_sentence(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and cleaned[-1] in SENTENCE_BREAK_CHARS
