"""Bounded workers whose callers can cancel pending workflow steps."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
from typing import Callable

from .cancellation import CancellationToken


class WorkflowWorkers:
    def __init__(self, capacity: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="support-stream")
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._tokens: set[CancellationToken] = set()
        self._closed = False

    def submit(self, token: CancellationToken, work: Callable[[], None]) -> Future:
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise RuntimeError("stream capacity reached; retry when another request finishes")
            self._tokens.add(token)
            try:
                future = self._executor.submit(work)
            except Exception:
                self._tokens.remove(token)
                self._slots.release()
                raise

        def finished(_: Future) -> None:
            with self._lock:
                self._tokens.remove(token)
                self._slots.release()

        future.add_done_callback(finished)
        return future

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for token in self._tokens:
                token.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
