"""toon_ai.core.logger"""

from __future__ import annotations

from typing import Literal, TypeAlias
import logging

from rich.logging import RichHandler


LoggerVerbosity: TypeAlias = Literal["verbose", "debug"] | str


_TOON_AI_ROOT_LOGGER: logging.Logger | None = None


_TOON_AI_LOG_LEVEL = "NOTSET"


def set_logger_verbosity(verbosity: LoggerVerbosity | None = None) -> None:
    """Set the verbosity of the logger."""
    global _TOON_AI_LOG_LEVEL, _TOON_AI_ROOT_LOGGER

    if not verbosity:
        verbosity_level = "warning"
    elif verbosity == "verbose":
        verbosity_level = "info"
    elif verbosity == "debug":
        verbosity_level = "debug"
    else:
        verbosity_level = verbosity

    _TOON_AI_LOG_LEVEL = verbosity_level
    _TOON_AI_ROOT_LOGGER = _configure_root_logger(_TOON_AI_LOG_LEVEL)


def _configure_root_logger(level: str) -> logging.Logger:
    """Configure the root toon_ai logger with RichHandler."""
    numeric_level = getattr(logging, level.upper(), logging.NOTSET)

    logger = logging.getLogger("toon_ai")
    logger.setLevel(numeric_level)

    # Clear existing handlers we control to avoid duplicates
    logger.handlers = []

    handler = RichHandler(
        rich_tracebacks=True,
        markup=True,
        show_path=False,
    )
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    global _TOON_AI_ROOT_LOGGER
    if not _TOON_AI_ROOT_LOGGER:
        _TOON_AI_ROOT_LOGGER = logger
    else:
        _TOON_AI_ROOT_LOGGER.handlers = []
        _TOON_AI_ROOT_LOGGER.addHandler(handler)

    return logger


def _log_info_panel(
    logger: logging.Logger, title: str, lines: list[str] | None = None
) -> None:
    """Log an info panel with Rich."""

    if lines:
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f"  [turquoise2]>>>[/turquoise2] {line}")
        lines = formatted_lines

    if lines:
        logger.info(
            "[dim]-----------------------[/dim]\n"
            f"[bold]{title}[/bold]\n\n"
            + "\n".join(lines)
            + "\n[dim]-----------------------[/dim]\n\n"
        )
    else:
        logger.info(
            f"[dim]-----------------------[/dim]\n"
            f"[bold]{title}[/bold]\n"
            f"[dim]-----------------------[/dim]\n\n"
        )


def _log_debug_context(logger: logging.Logger, lines: list[str]) -> None:
    """Log a debug context with Rich."""
    logger.debug("[dim]" + "\n".join(lines) + "[/dim]")


_configure_root_logger("WARNING")
