from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from fabricated.captions.backends.base import CaptionBackend, CaptionBackendError
from fabricated.captions.types import Transcript


class WhisperXCaptionBackend(CaptionBackend):
    name = "whisperx"

    def transcribe(self, audio_path: Path) -> Transcript:
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise CaptionBackendError(
                "whisperx captioning requires `uv` in PATH so it can run in an isolated environment."
            )

        runner_path = Path(__file__).resolve().parents[1] / "runners" / "whisperx_runner.py"
        with tempfile.NamedTemporaryFile(prefix="fabricated-whisperx-", suffix=".json", delete=False) as handle:
            output_json = Path(handle.name)

        command = [
            uv_path,
            "run",
            "--isolated",
            "--with",
            "whisperx==3.8.4",
            "python",
            str(runner_path),
            "--input",
            str(audio_path),
            "--output",
            str(output_json),
            "--model",
            self.model_name,
        ]
        if self.config.language != "auto":
            command.extend(["--language", self.config.language])

        completed = subprocess.run(command, capture_output=True, text=True)
        try:
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or completed.stdout.strip()
                raise CaptionBackendError(
                    "whisperx transcription failed. "
                    "This backend is run in an isolated environment because its published dependency stack "
                    "conflicts with the audio-enhancement stack. "
                    f"Details: {stderr}"
                )

            with output_json.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return Transcript.from_dict(payload)
        finally:
            output_json.unlink(missing_ok=True)
