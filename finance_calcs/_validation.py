"""Shared validation for public calculation inputs."""

from __future__ import annotations

import math


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
