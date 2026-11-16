#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the render/pipeline scheduling helpers (node_editor/scheduling.py)."""
import importlib.util
import os
import threading
import time

import pytest

# Loaded from its file: several test modules replace the ``node_editor``
# package in sys.modules with a stub at collection time.
_spec = importlib.util.spec_from_file_location(
    "_node_editor_scheduling_under_test",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "node_editor",
        "scheduling.py",
    ),
)
util = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(util)


def _run_unlocked(lock, func, *args):
    """What node_editor.util.run_unlocked does with the global lock."""
    with lock.released():
        return func(*args)


def test_render_thread_does_not_wait_for_a_whole_pipeline_pass():
    """The worker re-takes the lock once per node without ever sleeping."""
    lock = util.UiPriorityLock()
    node_time = 0.01
    nodes = 30
    started = threading.Event()

    def pipeline_pass():
        for _ in range(nodes):
            with lock:
                started.set()
                time.sleep(node_time)

    worker = threading.Thread(target=pipeline_pass)
    worker.start()
    started.wait(2)

    begin = time.perf_counter()
    with lock.ui_frame():
        waited = time.perf_counter() - begin
    worker.join()

    # At most the node being updated and one already queued behind it,
    # not the 30 nodes of the pass.
    assert waited < 5 * node_time


def test_worker_is_not_starved_by_the_render_loop():
    lock = util.UiPriorityLock()
    stop = threading.Event()
    updates = []

    def render_loop():
        while not stop.is_set():
            with lock.ui_frame():
                time.sleep(0.002)
            time.sleep(0.001)

    ui = threading.Thread(target=render_loop)
    ui.start()
    try:
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            with lock:
                updates.append(1)
    finally:
        stop.set()
        ui.join()

    assert len(updates) > 20


def test_lock_is_reentrant_and_supports_timeouts():
    lock = util.UiPriorityLock()
    with lock:
        with lock:
            with lock.ui_frame():
                pass
        result = []
        other = threading.Thread(target=lambda: result.append(lock.acquire(timeout=0.05)))
        other.start()
        other.join()
        assert result == [False]

    assert lock.acquire(timeout=0.05) is True
    lock.release()
    with pytest.raises(RuntimeError):
        lock.release()


def test_released_frees_the_lock_and_takes_it_back():
    lock = util.UiPriorityLock()
    seen = []

    def compute(value):
        other = threading.Thread(
            target=lambda: seen.append(lock.acquire(timeout=1) and lock.release() is None)
        )
        other.start()
        other.join()
        return value * 2

    with lock:
        with lock:
            assert _run_unlocked(lock, compute, 21) == 42
            # Held again, at the same depth: another thread is kept out
            blocked = []
            other = threading.Thread(target=lambda: blocked.append(lock.acquire(timeout=0.05)))
            other.start()
            other.join()
            assert blocked == [False]
    assert seen == [True]
    with pytest.raises(RuntimeError):
        lock.release()

    # Without the lock (synchronous mode) it is a plain call
    assert _run_unlocked(lock, compute, 1) == 2


def test_released_takes_the_lock_back_when_the_computation_fails():
    lock = util.UiPriorityLock()

    def fail():
        raise ValueError("inference failed")

    with lock:
        with pytest.raises(ValueError):
            _run_unlocked(lock, fail)
        lock.release()
        lock.acquire()


class _FakeTime:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(round(delay, 6))
        self.now += delay


def test_frame_pacer_waits_for_the_rest_of_the_frame():
    fake = _FakeTime()
    pacer = util.FramePacer(fps=50, clock=fake.clock, sleep=fake.sleep)

    for render_time in (0.004, 0.012, 0.0):
        fake.now += render_time
        pacer.wait()

    assert fake.sleeps == [0.016, 0.008, 0.02]


def test_frame_pacer_always_yields_and_does_not_catch_up_after_a_slow_frame():
    fake = _FakeTime()
    pacer = util.FramePacer(fps=50, clock=fake.clock, sleep=fake.sleep)

    fake.now += 0.2  # a frame ten times too long
    pacer.wait()
    fake.now += 0.004
    pacer.wait()

    assert fake.sleeps == [pacer.MIN_IDLE, 0.016]
