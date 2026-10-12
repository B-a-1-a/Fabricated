from fabricated.config import PipelineConfig
from fabricated.media.ffmpeg import build_filter_chain, build_subtitle_filter


def test_build_filter_chain_contains_expected_filters():
    config = PipelineConfig.from_options(preset="studio")
    chain = build_filter_chain(config)
    assert "highpass=" in chain
    assert "deesser=" in chain
    assert "acompressor=" in chain
    assert "speechnorm=" in chain
    assert "loudnorm=" in chain
    assert "alimiter=" in chain


def test_fast_preset_uses_different_filter_values():
    studio = build_filter_chain(PipelineConfig.from_options(preset="studio"))
    fast = build_filter_chain(PipelineConfig.from_options(preset="fast"))
    assert studio != fast


def test_build_subtitle_filter_contains_expected_parts(tmp_path):
    subtitle = tmp_path / "captions.ass"
    fonts_dir = tmp_path / "fonts"
    filter_value = build_subtitle_filter(subtitle, fonts_dir, width=1280, height=720)
    assert "ass=" in filter_value
    assert "fontsdir=" in filter_value
    assert "original_size=1280x720" in filter_value
