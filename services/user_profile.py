"""Validation helpers for user-provided profile fields."""

from __future__ import annotations

import re


USER_NAME_ERROR = "Nami kedah 2 dugi 30 aksara, tanpa tautan atanapi simbol anéh."
_URL_LIKE_NAME_RE = re.compile(
    r"(?i)(?:https?://|www\.|(?:^|\s)[a-z0-9-]+(?:\.[a-z0-9-]+)+\b)"
)


def validate_user_name(value: object) -> str:
    """Return a trimmed, safe display name or raise a user-facing ValueError."""
    name = value.strip() if isinstance(value, str) else ""
    if not 2 <= len(name) <= 30:
        raise ValueError(USER_NAME_ERROR)
    if _URL_LIKE_NAME_RE.search(name):
        raise ValueError(USER_NAME_ERROR)
    if not all(character.isalnum() or character in " .-" for character in name):
        raise ValueError(USER_NAME_ERROR)
    return name


def needs_user_name(uid_final: bool, value: object) -> bool:
    """Only request a name after a persistent device identity is established."""
    return uid_final and not (isinstance(value, str) and value.strip())
