from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fabricated.config import PipelineConfig
from fabricated.media.youtube import SUPPORTED_AUDIO_FORMATS, download_youtube_media
from fabricated.pipeline.process_audio import enhance_audio
from fabricated.pipeline.process_video import process_video
from fabricated.providers.base import ProviderError


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        if args.command == "process":
            config = _build_config(args)
            result = process_video(args.input, args.out, config)
        elif args.command == "enhance-audio":
            config = _build_config(args)
            result = enhance_audio(args.input, args.out, config)
        elif args.command == "download-youtube":
            output = download_youtube_media(
                args.url,
                args.out,
                extract_audio=args.extract_audio,
                audio_format=args.audio_format,
            )
            logging.info("%s: %s", "Extracted audio" if args.extract_audio else "Downloaded video", output)
            return 0
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except (OSError, ValueError, ProviderError, RuntimeError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    if result.video_output is not None:
        logging.info("Video output: %s", result.video_output)
    logging.info("Audio output: %s", result.audio_output)
    if result.subtitle_output is not None:
        logging.info("Subtitle output: %s", result.subtitle_output)
    if result.transcript_output is not None:
        logging.info("Transcript output: %s", result.transcript_output)
    logging.info("Provider: %s", result.provider)
    logging.info("Sample rate: %s Hz", result.sample_rate)
    logging.info("Duration: %.2f s", result.duration_sec)
    if result.temp_dir is not None:
        logging.info("Temp dir: %s", result.temp_dir)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fabricated")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Clean a video and remux the result.")
    _add_common_arguments(process_parser)
    process_parser.add_argument("input", type=Path, help="Input video path.")
    process_parser.add_argument("--out", type=Path, help="Output MP4 path. Defaults to <input>.cleaned.mp4.")
    process_parser.add_argument(
        "--no-remux",
        action="store_true",
        help="Skip MP4 creation and export only the cleaned WAV.",
    )
    process_parser.add_argument("--captions", action="store_true", help="Generate high-impact synced captions.")
    process_parser.add_argument(
        "--caption-backend",
        default="faster-whisper",
        choices=("faster-whisper", "whisperx"),
        help="Caption transcription backend.",
    )
    process_parser.add_argument(
        "--caption-style",
        default="impact",
        choices=("highlight", "zoom", "impact"),
        help="Current-word caption emphasis style.",
    )
    process_parser.add_argument(
        "--caption-font",
        default="impact",
        choices=("impact", "avenir-next-condensed", "futura", "helvetica-neue"),
        help="Caption font preset.",
    )
    process_parser.add_argument(
        "--caption-language",
        default="auto",
        help="Caption language, or 'auto' to detect.",
    )
    process_parser.add_argument(
        "--caption-model",
        help="Override the transcription model for the selected caption backend.",
    )

    audio_parser = subparsers.add_parser("enhance-audio", help="Clean an audio file and export WAV.")
    _add_common_arguments(audio_parser)
    audio_parser.add_argument("input", type=Path, help="Input audio path.")
    audio_parser.add_argument("--out", type=Path, help="Output WAV path. Defaults to <input>.cleaned.wav.")

    download_parser = subparsers.add_parser(
        "download-youtube",
        help="Download a YouTube video or extract its audio track.",
    )
    download_parser.add_argument("url", help="YouTube video URL.")
    download_parser.add_argument(
        "--out",
        type=Path,
        help="Output path or output directory. Defaults to the current directory.",
    )
    download_parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Download and convert only the audio track.",
    )
    download_parser.add_argument(
        "--audio-format",
        default="mp3",
        choices=SUPPORTED_AUDIO_FORMATS,
        help="Audio format used with --extract-audio.",
    )

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", default="studio", choices=("studio", "fast"))
    parser.add_argument("--provider", choices=("clearvoice", "deepfilternet"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    parser.add_argument("--bandwidth-extend", action="store_true")
    parser.add_argument(
        "--export-audio",
        action="store_true",
        default=None,
        help="Persist the cleaned WAV alongside the main output.",
    )
    parser.add_argument("--keep-temp", action="store_true")


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig.from_options(
        preset=args.preset,
        provider=args.provider,
        device=args.device,
        bandwidth_extend=args.bandwidth_extend,
        export_audio=args.export_audio,
        remux_video=not getattr(args, "no_remux", False),
        keep_temp=args.keep_temp,
        captions=getattr(args, "captions", False),
        caption_backend=getattr(args, "caption_backend", "faster-whisper"),
        caption_style=getattr(args, "caption_style", "impact"),
        caption_font=getattr(args, "caption_font", "impact"),
        caption_language=getattr(args, "caption_language", "auto"),
        caption_model=getattr(args, "caption_model", None),
    )


if __name__ == "__main__":
    raise SystemExit(main())
