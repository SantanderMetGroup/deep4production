"""
deep4production logging.

A single ``✻`` marker prefix per record (orange on INFO, yellow on WARNING,
red on ERROR), with a dim ``·`` for DEBUG. Output is colored when stderr is
a TTY and uncolored otherwise (clean SLURM / CI logs). Wraps Python's stdlib
``logging`` so users can adjust verbosity, redirect to a file, etc.

Typical use
-----------
At the top of a CLI entry point or notebook cell::

    from deep4production.utils.log import setup_logging
    setup_logging(level="INFO")

In any module::

    from deep4production.utils.log import get_logger
    log = get_logger("dataset")
    log.info("Saved store at %s", path)
    log.warning("No files matched pattern: %s", pattern)
"""

import os
import sys
import logging

# ─────────────────────────────────────────────────────────────────────────────
# Marker + color per level. Same ``✻`` glyph for INFO/WARNING/ERROR with the
# colour carrying the level signal — visually consistent, like Claude's UI.
# ─────────────────────────────────────────────────────────────────────────────
_MARKERS = {
    logging.DEBUG:    "·",
    logging.INFO:     "✻",
    logging.WARNING:  "✻",
    logging.ERROR:    "✻",
    logging.CRITICAL: "✻",
}

# 256-colour ANSI; falls through to basic 8-colour if the terminal can't do 256.
_COLORS = {
    logging.DEBUG:    "\033[2m",          # dim
    logging.INFO:     "\033[38;5;208m",   # orange
    logging.WARNING:  "\033[33m",         # yellow
    logging.ERROR:    "\033[31m",         # red
    logging.CRITICAL: "\033[1;31m",       # bold red
}
_RESET = "\033[0m"
_DIM   = "\033[2m"


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


class D4PFormatter(logging.Formatter):
    """One-line, marker-prefixed formatter. Optional module tag in dim text."""

    def __init__(self, color: bool = True, show_logger: bool = False):
        super().__init__()
        self.color       = color
        self.show_logger = show_logger

    def format(self, record: logging.LogRecord) -> str:
        marker = _MARKERS.get(record.levelno, "·")
        msg    = record.getMessage()

        # Optional ``[d4p.dataset]`` tag, dim-coloured.
        tag = ""
        if self.show_logger and record.name and record.name != "d4p":
            short = record.name.removeprefix("d4p.")
            tag   = f"{_DIM}[{short}]{_RESET} " if self.color else f"[{short}] "

        if self.color:
            color = _COLORS.get(record.levelno, "")
            line  = f"{color}{marker}{_RESET} {tag}{msg}"
        else:
            line = f"{marker} {tag}{msg}"

        # Append exception traceback if present, indented for readability.
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(level: str | int = "INFO",
                  color: str | bool = "auto",
                  show_logger: bool = False,
                  stream=None) -> logging.Logger:
    """
    Configure the ``d4p`` root logger. Idempotent — safe to call multiple
    times (replaces existing handlers).

    Parameters
    ----------
    level : str or int, default ``"INFO"``
        ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    color : ``"auto"``, ``True``, ``False``
        ``"auto"`` enables color when the output stream is a TTY (and
        ``NO_COLOR`` is unset).
    show_logger : bool
        Prefix each line with ``[<sub-logger>]`` (e.g. ``[dataset]``).
    stream : file-like, optional
        Defaults to ``sys.stderr``.
    """
    stream = stream or sys.stderr
    if color == "auto":
        color = _supports_color(stream)

    log = logging.getLogger("d4p")
    log.setLevel(level)
    log.propagate = False

    # Drop any handlers we previously installed so re-running setup_logging
    # doesn't double up output in notebooks.
    for h in list(log.handlers):
        log.removeHandler(h)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(D4PFormatter(color=bool(color), show_logger=show_logger))
    log.addHandler(handler)
    return log


def get_logger(name: str) -> logging.Logger:
    """Return ``logging.getLogger("d4p." + name)``."""
    return logging.getLogger(f"d4p.{name}")
