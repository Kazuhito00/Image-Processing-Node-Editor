#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sharing of the DearPyGui lock between the render loop and the pipeline.

Standard library only: this module must stay importable without DearPyGui.
"""
import threading
import time
from contextlib import contextmanager


class UiPriorityLock:
    """Reentrant lock in which the render thread never queues behind workers.

    A plain ``threading.RLock`` is not fair: a worker that releases the lock
    and takes it again right away (the pipeline loop does so once per node)
    keeps it for as long as it wants, and the UI only gets a frame in when the
    worker happens to sleep.  Here the render thread announces itself with
    :meth:`ui_frame`; every other thread then waits for the frame to be drawn
    before taking the lock again.  The UI therefore never waits for more than
    the work already in progress.

    Drop-in replacement for ``threading.RLock`` (``with``, ``acquire``,
    ``release``).
    """

    # Upper bound on how long a worker defers to the UI.  A frame is drawn in
    # a few milliseconds; this only guards against a render thread that died
    # or hung between announcing itself and releasing the lock.
    _UI_YIELD_TIMEOUT = 0.25

    def __init__(self):
        self._lock = threading.RLock()
        self._ui_idle = threading.Event()
        self._ui_idle.set()
        self._owner = None
        self._depth = 0

    def acquire(self, blocking=True, timeout=-1):
        # A thread that already owns the lock must never wait for the UI:
        # the UI is waiting for that very thread.
        if blocking and self._owner != threading.get_ident():
            self._ui_idle.wait(self._UI_YIELD_TIMEOUT)
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._owner = threading.get_ident()
            self._depth += 1
        return acquired

    def release(self):
        if self._owner != threading.get_ident():
            raise RuntimeError("cannot release un-acquired lock")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    @contextmanager
    def ui_frame(self):
        """Hold the lock for one rendered frame, ahead of waiting workers."""
        self._ui_idle.clear()
        try:
            self._lock.acquire()
            self._owner = threading.get_ident()
            self._depth += 1
            try:
                yield self
            finally:
                self.release()
        finally:
            self._ui_idle.set()

    @contextmanager
    def released(self):
        """Give the lock up for the duration of a DearPyGui-free computation.

        No-op when the calling thread does not hold the lock.  The lock is
        taken back, at the same nesting depth, before the block exits.
        """
        ident = threading.get_ident()
        if self._owner != ident:
            yield
            return
        depth = self._depth
        self._depth = 0
        self._owner = None
        for _ in range(depth):
            self._lock.release()
        try:
            yield
        finally:
            self._ui_idle.wait(self._UI_YIELD_TIMEOUT)
            for _ in range(depth):
                self._lock.acquire()
            self._owner = ident
            self._depth = depth


class FramePacer:
    """Paces the render loop to a target frame rate *outside* of the lock.

    With vsync enabled, ``dpg.render_dearpygui_frame()`` blocks until the next
    monitor refresh while the DearPyGui lock is held, so the pipeline thread
    can only update one node per displayed frame.  The viewport is therefore
    created without vsync and the waiting is done here, with the lock free.
    """

    # Always leave the lock free for a moment, even when frames are late,
    # so that the pipeline cannot be starved by a slow renderer.
    MIN_IDLE = 0.001

    def __init__(self, fps=60, clock=time.perf_counter, sleep=time.sleep):
        self.interval = 1.0 / max(1.0, float(fps))
        self._clock = clock
        self._sleep = sleep
        self._deadline = clock() + self.interval

    def wait(self):
        """Sleep until the next frame is due; returns the time slept."""
        now = self._clock()
        delay = self._deadline - now
        if delay < self.MIN_IDLE:
            delay = self.MIN_IDLE
            # Late: restart the schedule instead of rendering a burst of
            # frames to catch up.
            self._deadline = now + delay
        self._deadline += self.interval
        self._sleep(delay)
        return delay
