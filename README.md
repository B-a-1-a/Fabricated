# Fabricated

Offline video voice isolation for Apple Silicon using open-source speech enhancement backends.

## What it does

- downloads YouTube videos to local MP4 files or extracts audio-only exports
- extracts audio from a video or audio input
- normalizes it to 48 kHz mono WAV
- runs open-source speech enhancement
- applies FFmpeg cleanup filters for a crisper voice
- exports a cleaned WAV
- remuxes the cleaned audio back into an MP4 for video inputs

## Backends

- `clearvoice` for the default `studio` preset
- `deepfilternet` for the `fast` preset

## Setup

This project is pinned to Python 3.11 because the audio model stack is not targeted at the system Python in this workspace.

```bash
uv python install 3.11
uv sync --python 3.11
```

## Usage

```bash
uv run --python 3.11 fabricated download-youtube "https://www.youtube.com/watch?v=VIDEO_ID" --out downloads/source.mp4
uv run --python 3.11 fabricated download-youtube "https://www.youtube.com/watch?v=VIDEO_ID" --extract-audio --audio-format wav --out downloads/source.wav
uv run --python 3.11 fabricated process input.mov --out output.mp4
uv run --python 3.11 fabricated enhance-audio input.wav --out output.wav --preset fast
```

Use `download-youtube` only for videos you have the rights or permission to download.
