from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

FALLBACK_FONT_NAME = "helvetica-neue"
FONT_PRESETS: dict[str, dict[str, object]] = {
    "impact": {
        "family": "Impact",
        "paths": [
            Path("/System/Library/Fonts/Supplemental/Impact.ttf"),
        ],
    },
    "avenir-next-condensed": {
        "family": "Avenir Next Condensed",
        "paths": [
            Path("/System/Library/Fonts/Avenir Next Condensed.ttc"),
        ],
    },
    "futura": {
        "family": "Futura",
        "paths": [
            Path("/System/Library/Fonts/Supplemental/Futura.ttc"),
        ],
    },
    "helvetica-neue": {
        "family": "Helvetica Neue",
        "paths": [
            Path("/System/Library/Fonts/HelveticaNeue.ttc"),
            Path("/System/Library/Fonts/Helvetica.ttc"),
        ],
    },
}


@dataclass(slots=True, frozen=True)
class ResolvedFontPreset:
    requested_name: str
    resolved_name: str
    family_name: str
    source_path: Path
    fallback_used: bool = False


@dataclass(slots=True, frozen=True)
class PreparedFont:
    requested_name: str
    resolved_name: str
    family_name: str
    source_path: Path
    copied_path: Path
    fonts_dir: Path
    fallback_used: bool = False


def resolve_font_preset(name: str) -> ResolvedFontPreset:
    if name not in FONT_PRESETS:
        supported = ", ".join(sorted(FONT_PRESETS))
        raise ValueError(f"Unsupported caption font {name!r}. Expected one of: {supported}.")

    preset = FONT_PRESETS[name]
    source_path = _first_existing_path(preset["paths"])
    if source_path is not None:
        return ResolvedFontPreset(
            requested_name=name,
            resolved_name=name,
            family_name=str(preset["family"]),
            source_path=source_path,
            fallback_used=False,
        )

    fallback = FONT_PRESETS[FALLBACK_FONT_NAME]
    fallback_path = _first_existing_path(fallback["paths"])
    if fallback_path is None:
        raise FileNotFoundError("Unable to resolve a usable system font for captions.")

    return ResolvedFontPreset(
        requested_name=name,
        resolved_name=FALLBACK_FONT_NAME,
        family_name=str(fallback["family"]),
        source_path=fallback_path,
        fallback_used=True,
    )


def prepare_font(font: ResolvedFontPreset, fonts_dir: Path) -> PreparedFont:
    fonts_dir.mkdir(parents=True, exist_ok=True)
    copied_path = fonts_dir / font.source_path.name
    # copy2() tries to preserve macOS file flags from system fonts and can fail
    # with EPERM; a plain file copy is enough for ffmpeg's fontsdir lookup.
    shutil.copyfile(font.source_path, copied_path)
    return PreparedFont(
        requested_name=font.requested_name,
        resolved_name=font.resolved_name,
        family_name=font.family_name,
        source_path=font.source_path,
        copied_path=copied_path,
        fonts_dir=fonts_dir,
        fallback_used=font.fallback_used,
    )


def _first_existing_path(paths: object) -> Path | None:
    for path in paths:  # type: ignore[assignment]
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None
