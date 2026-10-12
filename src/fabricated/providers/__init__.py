from __future__ import annotations

from importlib import import_module

from fabricated.config import PipelineConfig
from fabricated.providers.base import AudioProvider

PROVIDER_ENTRYPOINTS: dict[str, tuple[str, str]] = {
    "clearvoice": ("fabricated.providers.clearvoice", "ClearVoiceProvider"),
    "deepfilternet": ("fabricated.providers.deepfilternet", "DeepFilterNetProvider"),
}


def resolve_provider_class(name: str) -> type[AudioProvider]:
    try:
        module_name, class_name = PROVIDER_ENTRYPOINTS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(PROVIDER_ENTRYPOINTS))
        raise ValueError(f"Unsupported provider {name!r}. Expected one of: {supported}.") from exc
    module = import_module(module_name)
    return getattr(module, class_name)


def get_provider(config: PipelineConfig) -> AudioProvider:
    provider_class = resolve_provider_class(config.provider)
    return provider_class(config)
