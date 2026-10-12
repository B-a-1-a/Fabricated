from fabricated.config import PipelineConfig, load_preset, validate_provider_name


def test_load_preset_reads_yaml():
    preset = load_preset("studio")
    assert preset["provider"] == "clearvoice"
    assert preset["provider_model"] == "MossFormer2_SE_48K"


def test_pipeline_config_merges_preset_and_overrides():
    config = PipelineConfig.from_options(
        preset="fast",
        provider="clearvoice",
        device="cpu",
        bandwidth_extend=True,
        keep_temp=True,
        captions=True,
        caption_backend="faster-whisper",
        caption_style="impact",
        caption_font="impact",
    )
    assert config.preset == "fast"
    assert config.provider == "clearvoice"
    assert config.device == "cpu"
    assert config.bandwidth_extend is True
    assert config.keep_temp is True
    assert config.post_filter is True
    assert config.captions is not None
    assert config.captions.backend == "faster-whisper"
    assert config.captions.style == "impact"


def test_validate_provider_name_rejects_unknown_provider():
    try:
        validate_provider_name("unknown")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected validate_provider_name() to fail for unknown providers.")
