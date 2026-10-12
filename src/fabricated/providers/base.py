from __future__ import annotations

import contextlib
import os
import platform
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from fabricated.config import DeviceName, PipelineConfig


class ProviderError(RuntimeError):
    """Raised when a provider cannot run or is unavailable."""


@dataclass(slots=True, frozen=True)
class ProviderResult:
    output_audio: Path
    sample_rate: int
    device_used: str
    notes: tuple[str, ...] = field(default_factory=tuple)


class AudioProvider(ABC):
    name: str

    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def enhance(self, input_audio: Path, output_audio: Path) -> ProviderResult:
        """Enhance the input audio and write to output_path."""


def resolve_device_preference(
    *,
    requested: DeviceName,
    provider_name: str,
    supports_mps: bool,
) -> tuple[str, tuple[str, ...]]:
    notes: list[str] = []
    if requested == "cpu":
        return "cpu", ()
    if requested == "mps":
        if supports_mps:
            return "mps", ()
        notes.append(f"{provider_name} does not expose stable MPS execution; falling back to CPU.")
        return "cpu", tuple(notes)
    if platform.system() == "Darwin" and not supports_mps:
        notes.append(f"{provider_name} will run on CPU on Apple Silicon.")
        return "cpu", tuple(notes)
    return "auto", ()


@contextlib.contextmanager
def temporary_environment(overrides: dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
