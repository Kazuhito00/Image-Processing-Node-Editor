#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the fast path of the YouTube node: a single yt-dlp extraction is
performed and reused for both the video and the audio format selection.
"""
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.modules['dearpygui'] = mock.MagicMock()
sys.modules['dearpygui.dearpygui'] = mock.MagicMock()

import pytest

import node.InputNode.node_youtube as yt
from node.InputNode.node_youtube import YoutubeNode


FAKE_INFO = {
    "url": "https://example.com/default",
    "formats": [
        {"url": "https://example.com/audio_low.webm", "vcodec": "none", "acodec": "opus", "ext": "webm", "abr": 48},
        {"url": "https://example.com/audio_m4a", "vcodec": "none", "acodec": "mp4a", "ext": "m4a", "abr": 128},
        {"url": "https://example.com/audio_hi.webm", "vcodec": "none", "acodec": "opus", "ext": "webm", "abr": 160},
        {"url": "https://example.com/720.mp4", "vcodec": "avc1", "acodec": "none", "height": 720},
        {"url": "https://example.com/360.m3u8", "vcodec": "avc1", "acodec": "mp4a", "height": 360},
        {"url": "https://example.com/480.mp4", "vcodec": "avc1", "acodec": "none", "height": 480},
        {"url": "https://example.com/240.mp4", "vcodec": "avc1", "acodec": "none", "height": 240},
    ],
}


def test_select_video_urls_prefers_low_direct_then_hls():
    urls = yt._select_video_urls(FAKE_INFO)
    assert urls == [
        "https://example.com/240.mp4",
        "https://example.com/480.mp4",
        "https://example.com/360.m3u8",
    ]


def test_select_audio_url_prefers_best_webm():
    assert yt._select_audio_url(FAKE_INFO) == "https://example.com/audio_hi.webm"
    assert yt._select_audio_url(None) is None
    assert yt._select_audio_url({"url": "x", "formats": []}) == "x"


def test_open_youtube_stream_single_extraction(monkeypatch):
    extract_calls = []

    def fake_extract(url, cookies_browser=None):
        extract_calls.append((url, cookies_browser))
        return FAKE_INFO

    opened = []

    def fake_open_capture(video_url):
        opened.append(video_url)
        return mock.MagicMock() if video_url.endswith("480.mp4") else None

    monkeypatch.setattr(yt, "_extract_info", fake_extract)
    monkeypatch.setattr(yt, "_open_capture", fake_open_capture)
    monkeypatch.setattr(yt, "_open_with_streamlink", lambda url: pytest.fail("streamlink should not be used"))

    cap, info = yt.open_youtube_stream("https://youtube.com/watch?v=abc")

    assert cap is not None
    assert info is FAKE_INFO
    assert extract_calls == [("https://youtube.com/watch?v=abc", None)]
    assert opened == ["https://example.com/240.mp4", "https://example.com/480.mp4"]


def test_open_youtube_stream_retries_with_cookies_on_bot_detection(monkeypatch):
    extract_calls = []

    def fake_extract(url, cookies_browser=None):
        extract_calls.append(cookies_browser)
        if cookies_browser is None:
            raise RuntimeError("Sign in to confirm you're not a bot")
        return FAKE_INFO

    monkeypatch.setattr(yt, "_extract_info", fake_extract)
    monkeypatch.setattr(yt, "_open_capture", lambda url: mock.MagicMock())
    monkeypatch.setattr(yt, "_open_with_streamlink", lambda url: None)

    cap, info = yt.open_youtube_stream("https://youtube.com/watch?v=abc", cookies_browser="firefox")

    assert cap is not None
    assert extract_calls == [None, "firefox"]


def test_open_youtube_stream_falls_back_to_streamlink(monkeypatch):
    monkeypatch.setattr(yt, "_extract_info", lambda url, cookies_browser=None: {"formats": []})
    monkeypatch.setattr(yt, "_open_with_streamlink", lambda url: mock.MagicMock())

    cap, info = yt.open_youtube_stream("https://youtube.com/watch?v=abc")

    assert cap is not None
    assert info is None


def test_audio_url_reuses_stream_info_without_extraction(monkeypatch):
    node = YoutubeNode()
    node._stream_info = FAKE_INFO

    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", lambda *a, **k: pytest.fail("no extraction expected"))

    assert node._get_audio_stream_url("https://youtube.com/watch?v=abc") == "https://example.com/audio_hi.webm"
