"""
resonait/claw/__init__.py
==========================
ResonAItClaw — Active AGI Agent Package
"""

from resonait.claw.claw import ResonAItClaw, ClawConfig, ClawMessage
from resonait.claw.emotion_engine import (
    EmotionEngine,
    InitiativeEngine,
    EnvironmentObserver,
    EmotionType,
    EmotionalState,
)

__all__ = [
    "ResonAItClaw",
    "ClawConfig",
    "ClawMessage",
    "EmotionEngine",
    "InitiativeEngine",
    "EnvironmentObserver",
    "EmotionType",
    "EmotionalState",
]
