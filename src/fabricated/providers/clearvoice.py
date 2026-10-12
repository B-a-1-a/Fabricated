from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import soundfile as sf

from fabricated.providers.base import AudioProvider, ProviderError, ProviderResult, resolve_device_preference


class ClearVoiceProvider(AudioProvider):
    name = "clearvoice"

    def enhance(self, input_audio: Path, output_audio: Path) -> ProviderResult:
        try:
            import torch.backends.mps as torch_mps
            from clearvoice import ClearVoice
        except ImportError as exc:
            raise ProviderError("clearvoice is not installed in the active environment.") from exc

        device_used, notes = resolve_device_preference(
            requested=self.config.device,
            provider_name=self.name,
            supports_mps=False,
        )

        with _clearvoice_device_override(device_used, torch_mps):
            enhancement = ClearVoice(
                task="speech_enhancement",
                model_names=[self.config.provider_model or "MossFormer2_SE_48K"],
            )
            audio, sample_rate = sf.read(input_audio, dtype="float32")
            batched = _ensure_batch(audio)
            enhanced = enhancement(batched, False)
            enhanced_audio = _unwrap_batch(enhanced)

            if self.config.bandwidth_extend:
                if not self.config.super_resolution_model:
                    raise ProviderError("bandwidth extension requested, but no super-resolution model is configured.")
                super_resolution = ClearVoice(
                    task="speech_super_resolution",
                    model_names=[self.config.super_resolution_model],
                )
                enhanced_audio = _unwrap_batch(super_resolution(_ensure_batch(enhanced_audio), False))

        output_audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_audio, enhanced_audio, sample_rate, subtype="FLOAT")
        return ProviderResult(
            output_audio=output_audio,
            sample_rate=sample_rate,
            device_used=device_used,
            notes=notes,
        )


@contextlib.contextmanager
def _clearvoice_device_override(device_used: str, torch_mps):
    if device_used != "cpu":
        yield
        return

    original_is_available = torch_mps.is_available
    try:
        torch_mps.is_available = lambda: False
        yield
    finally:
        torch_mps.is_available = original_is_available


def _ensure_batch(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.reshape(1, -1).astype(np.float32)
    if audio.ndim == 2 and audio.shape[0] == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        mono = audio.mean(axis=1, dtype=np.float32)
        return mono.reshape(1, -1)
    raise ProviderError(f"Unsupported audio shape for ClearVoice: {audio.shape!r}")


def _unwrap_batch(audio: object) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim >= 2:
        return array[0]
    raise ProviderError("ClearVoice returned an empty tensor-like result.")
