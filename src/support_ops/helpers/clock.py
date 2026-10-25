"""Injectable clock. Nothing in the system calls datetime.now() directly, so return
windows, SLA timers and token expiry can be replayed exactly in simulated time."""

from __future__ import annotations

from datetime import datetime, timedelta


class Clock:
    def __init__(self, start: datetime):
        if start.tzinfo is None:
            raise ValueError("clock start must be timezone-aware")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now

    def set(self, when: datetime) -> None:
        self._now = when
