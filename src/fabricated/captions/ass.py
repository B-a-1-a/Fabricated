from __future__ import annotations

from dataclasses import dataclass

from fabricated.captions.chunking import NO_SPACE_PREFIXES
from fabricated.captions.fonts import PreparedFont
from fabricated.captions.types import CaptionChunk

PRIMARY_COLOR = "&H00FFFFFF"
ACTIVE_COLOR = "&H0000CCFF"
OUTLINE_COLOR = "&H00000000"
BACK_COLOR = "&H64000000"


@dataclass(slots=True, frozen=True)
class CaptionStylePreset:
    name: str
    active_scale: float
    active_color: str | None


STYLE_PRESETS: dict[str, CaptionStylePreset] = {
    "highlight": CaptionStylePreset(name="highlight", active_scale=1.0, active_color=ACTIVE_COLOR),
    "zoom": CaptionStylePreset(name="zoom", active_scale=1.28, active_color=None),
    "impact": CaptionStylePreset(name="impact", active_scale=1.28, active_color=ACTIVE_COLOR),
}


def render_ass_script(
    chunks: list[CaptionChunk],
    font: PreparedFont,
    style_name: str,
    *,
    video_width: int,
    video_height: int,
) -> str:
    style = STYLE_PRESETS[style_name]
    base_font_size, side_margin, bottom_margin = _caption_layout(video_width, video_height)
    active_font_size = int(base_font_size * style.active_scale)
    header = _build_header(
        font.family_name,
        base_font_size,
        video_width,
        video_height,
        side_margin,
        bottom_margin,
    )
    dialogue_lines = [
        _dialogue_line(start, end, _render_chunk_text(chunk, index, style, active_font_size))
        for chunk in chunks
        for index, start, end in _iter_chunk_events(chunk)
    ]
    body = "\n".join(dialogue_lines)
    return f"{header}\n{body}\n"


def _build_header(
    font_family: str,
    base_font_size: int,
    width: int,
    height: int,
    side_margin: int,
    bottom_margin: int,
) -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 0",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_family},{base_font_size},{PRIMARY_COLOR},{ACTIVE_COLOR},{OUTLINE_COLOR},{BACK_COLOR},-1,0,0,0,100,100,0,0,1,4,0,2,{side_margin},{side_margin},{bottom_margin},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )


def _caption_layout(width: int, height: int) -> tuple[int, int, int]:
    min_dim = min(width, height)
    portrait = height > width
    base_font_size = max(28, min(int(min_dim * 0.07), 88))
    side_margin = max(int(width * (0.08 if portrait else 0.06)), 48)
    bottom_margin = max(int(height * (0.14 if portrait else 0.10)), 72)
    return base_font_size, side_margin, bottom_margin


def _iter_chunk_events(chunk: CaptionChunk):
    for index, word in enumerate(chunk.words):
        start = max(chunk.start, word.start)
        next_start = chunk.words[index + 1].start if index + 1 < len(chunk.words) else chunk.end
        end = max(word.end, next_start, start + 0.02)
        end = min(max(end, start + 0.02), max(chunk.end, start + 0.02))
        yield index, start, end


def _render_chunk_text(
    chunk: CaptionChunk,
    active_index: int,
    style: CaptionStylePreset,
    active_font_size: int,
) -> str:
    rendered: list[str] = []
    for index, word in enumerate(chunk.words):
        token = _escape_ass_text(word.text.strip())
        if not token:
            continue
        token = _apply_active_style(token, style, active_font_size) if index == active_index else token
        if not rendered:
            rendered.append(token)
        elif word.text.strip().startswith("'") or word.text.strip()[:1] in NO_SPACE_PREFIXES:
            rendered[-1] = f"{rendered[-1]}{token}"
        else:
            rendered.append(token)
    return " ".join(rendered)


def _apply_active_style(token: str, style: CaptionStylePreset, active_font_size: int) -> str:
    tags = [f"\\fs{active_font_size}"] if style.active_scale != 1.0 else []
    if style.active_color:
        tags.append(f"\\1c{style.active_color}")
    if not tags:
        return token
    return "{%s}%s{\\rDefault}" % ("".join(tags), token)


def _dialogue_line(start: float, end: float, text: str) -> str:
    return f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"


def _ass_time(seconds: float) -> str:
    total_centiseconds = max(int(round(seconds * 100)), 0)
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
