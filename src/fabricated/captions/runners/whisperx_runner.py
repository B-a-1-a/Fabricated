from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    import whisperx

    device = "cpu"
    compute_type = "float32"
    language = args.language if args.language not in (None, "", "auto") else None

    audio = whisperx.load_audio(args.input)
    model = whisperx.load_model(args.model, device, compute_type=compute_type, language=language)
    result = model.transcribe(audio, batch_size=args.batch_size, language=language)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    segments = []
    full_text_parts: list[str] = []
    for segment in result.get("segments", []):
        words = []
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()
            start = word.get("start")
            end = word.get("end")
            if not text or start is None or end is None:
                continue
            words.append(
                {
                    "text": text,
                    "start": float(start),
                    "end": max(float(end), float(start) + 0.02),
                    "confidence": float(word["score"]) if word.get("score") is not None else None,
                }
            )

        text = str(segment.get("text", "")).strip()
        start = float(segment.get("start", 0.0))
        end = max(float(segment.get("end", start + 0.02)), start + 0.02)
        if not words and text:
            words = [{"text": text, "start": start, "end": end, "confidence": None}]
        if not words:
            continue

        segments.append(
            {
                "text": text or " ".join(word["text"] for word in words),
                "start": start,
                "end": end,
                "words": words,
            }
        )
        full_text_parts.append(text or " ".join(word["text"] for word in words))

    payload = {
        "backend": "whisperx",
        "language": result.get("language", language or "unknown"),
        "model": args.model,
        "text": " ".join(part for part in full_text_parts if part).strip(),
        "segments": segments,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
