from __future__ import annotations

from pathlib import Path

import soundfile as sf

from fabricated.providers.base import (
    AudioProvider,
    ProviderError,
    ProviderResult,
    resolve_device_preference,
    temporary_environment,
)


class DeepFilterNetProvider(AudioProvider):
    name = "deepfilternet"

    def enhance(self, input_audio: Path, output_audio: Path) -> ProviderResult:
        try:
            from df.enhance import enhance, init_df
            from df.io import load_audio
            import torch
        except ImportError as exc:
            raise ProviderError("deepfilternet is not installed in the active environment.") from exc

        device_used, notes = resolve_device_preference(
            requested=self.config.device,
            provider_name=self.name,
            supports_mps=False,
        )

        requested_device = "cpu" if device_used in {"cpu", "auto"} else device_used
        with temporary_environment({"DEVICE": requested_device}):
            model, df_state, _, _ = init_df(post_filter=self.config.post_filter, log_file=None)
            audio, _ = load_audio(str(input_audio), sr=df_state.sr())
            enhanced_audio = enhance(model, df_state, audio)

        waveform = torch.as_tensor(enhanced_audio).squeeze().detach().cpu().numpy()
        output_audio.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_audio, waveform, self.config.sample_rate, subtype="FLOAT")
        return ProviderResult(
            output_audio=output_audio,
            sample_rate=self.config.sample_rate,
            device_used=requested_device.lower(),
            notes=notes,
        )
