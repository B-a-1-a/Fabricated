from types import SimpleNamespace

from fabricated.config import PipelineConfig
from fabricated.providers import get_provider, resolve_provider_class
from fabricated.providers.clearvoice import _clearvoice_device_override


def test_get_provider_returns_named_provider():
    provider = get_provider(PipelineConfig.from_options(preset="fast"))
    assert provider.name == "deepfilternet"


def test_get_provider_rejects_unknown_provider():
    try:
        resolve_provider_class("bogus")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected provider resolution to fail.")


def test_clearvoice_cpu_override_disables_mps_temporarily():
    fake_mps = SimpleNamespace(is_available=lambda: True)
    original = fake_mps.is_available

    with _clearvoice_device_override("cpu", fake_mps):
        assert fake_mps.is_available() is False

    assert fake_mps.is_available is original
