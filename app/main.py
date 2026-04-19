"""Convenience entry point: ``python -m app.main`` is equivalent to ``python -m app.cli``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
