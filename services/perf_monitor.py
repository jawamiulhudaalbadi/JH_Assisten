"""Small process-local performance history for opted-in diagnostics."""

from __future__ import annotations

from collections import deque
import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_RECENT_RUNS: deque[dict[str, Any]] = deque(maxlen=30)
_RECENT_ADMIN_MENU_RUNS: deque[dict[str, Any]] = deque(maxlen=30)


def record_run(durations: dict[str, float], total_seconds: float) -> None:
    """Keep a bounded timing sample and warn about slow application reruns."""
    sample = {
        "total_ms": round(total_seconds * 1000, 1),
        **{
            f"{name}_ms": round(seconds * 1000, 1)
            for name, seconds in durations.items()
        },
    }
    with _LOCK:
        _RECENT_RUNS.append(sample)
    if total_seconds > 1.5:
        logger.warning(
            "Slow application rerun total_ms=%.1f identity_ms=%.1f "
            "configuration_ms=%.1f key_pool_ms=%.1f index_loading_ms=%.1f "
            "history_loading_ms=%.1f render_ms=%.1f",
            sample["total_ms"],
            sample.get("identity_ms", 0.0),
            sample.get("configuration_ms", 0.0),
            sample.get("key_pool_ms", 0.0),
            sample.get("index_loading_ms", 0.0),
            sample.get("history_loading_ms", 0.0),
            sample.get("render_ms", 0.0),
        )


def recent_runs() -> list[dict[str, Any]]:
    """Return a copy of the most recent bounded timing samples."""
    with _LOCK:
        return list(_RECENT_RUNS)


def record_admin_menu_run(menu: str, duration_seconds: float) -> None:
    """Keep the latest bounded dashboard-menu timings."""
    sample = {
        "menu": menu[:40],
        "duration_ms": round(max(0.0, duration_seconds) * 1000, 1),
    }
    with _LOCK:
        _RECENT_ADMIN_MENU_RUNS.append(sample)
    if duration_seconds > 1.5:
        logger.warning(
            "Slow admin dashboard menu menu=%s duration_ms=%.1f",
            sample["menu"],
            sample["duration_ms"],
        )


def recent_admin_menu_runs() -> list[dict[str, Any]]:
    with _LOCK:
        return list(_RECENT_ADMIN_MENU_RUNS)


def reset_recent_runs() -> None:
    """Clear process-local samples for isolated tests."""
    with _LOCK:
        _RECENT_RUNS.clear()
        _RECENT_ADMIN_MENU_RUNS.clear()
