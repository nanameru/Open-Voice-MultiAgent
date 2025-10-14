"""Shared dataclasses used by agent workflows."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CharacterData:
    """Structured information describing a story character."""

    name: Optional[str] = None
    background: Optional[str] = None


@dataclass
class StoryData:
    """Shared story context passed between agents via LiveKit."""

    characters: list[CharacterData] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    theme: Optional[str] = None
