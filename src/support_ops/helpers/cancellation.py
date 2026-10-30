"""Cooperative cancellation between synchronous workflow operations."""
from threading import Event


class WorkflowCancelled(Exception):
    """The caller no longer needs this workflow's result."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise WorkflowCancelled("workflow cancelled")
