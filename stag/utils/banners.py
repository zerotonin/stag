# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — utils.banners                                            ║
# ║  « .ljust()-aligned box-drawing banners for module headers »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Generate perfectly-aligned ASCII banners for module, section,   ║
# ║  and sub-section headers.  Use these helpers instead of hand-    ║
# ║  typing box-drawing glyphs — alignment is otherwise impossible   ║
# ║  to guarantee across editors and font metrics.                   ║
# ║                                                                  ║
# ║  Three hierarchy levels:                                         ║
# ║    1. double_banner()  — top-of-module identity                  ║
# ║    2. section_banner() — major sections within a module          ║
# ║    3. thin_rule()      — minor sub-sections                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Box-drawing banner generators for STAG module headers."""

from __future__ import annotations


def double_banner(
    package: str,
    module: str,
    tagline: str,
    description_lines: list[str],
    width: int = 66,
) -> str:
    """Generate a double-line module banner with perfect alignment.

    Args:
        package:           Package name (e.g. "STAG").
        module:            Module name (e.g. "clustering.internal_metrics").
        tagline:           Short « tagline » caption.
        description_lines: Body lines of the banner.
        width:             Inner width in characters (default 66).

    Returns:
        Multi-line string ready to paste as a comment block.
    """
    w = width
    lines: list[str] = []
    lines.append(f"# ╔{'═' * w}╗")
    lines.append(f"# ║  {package} — {module}".ljust(w + 2) + " ║")
    lines.append(f"# ║  « {tagline} »".ljust(w + 2) + " ║")
    lines.append(f"# ╠{'═' * w}╣")
    for desc in description_lines:
        lines.append(f"# ║  {desc}".ljust(w + 2) + " ║")
    lines.append(f"# ╚{'═' * w}╝")
    return "\n".join(lines)


def section_banner(title: str, tagline: str, width: int = 60) -> str:
    """Generate a single-line section header with perfect alignment."""
    w = width
    lines: list[str] = []
    lines.append(f"# ┌{'─' * w}┐")
    lines.append(f"# │ {title}  « {tagline} »".ljust(w + 2) + " │")
    lines.append(f"# └{'─' * w}┘")
    return "\n".join(lines)


def thin_rule(title: str, width: int = 65) -> str:
    """Generate a thin section separator."""
    lines: list[str] = []
    lines.append(f"# {'─' * width}")
    lines.append(f"#  {title}")
    lines.append(f"# {'─' * width}")
    return "\n".join(lines)
