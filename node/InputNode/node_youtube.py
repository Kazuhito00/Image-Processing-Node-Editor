#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import subprocess
import threading
import shutil
import queue

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import yt_dlp

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from node_editor.util import dpg_get_value, dpg_set_value
from node.node_abc import DpgNodeABC
from node.basenode import Node
from node.VideoNode.sync import FramePacket


def _get_ffmpeg_exe():
    """Return the path to a usable ffmpeg executable."""
    try:
        if imageio_ffmpeg is not None:
            return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _is_bot_detection_error(error_str):
    """Return True if the error looks like a YouTube bot/auth detection."""
    keywords = ["sign in", "confirm you're not a bot", "use --cookies"]
    error_lower = str(error_str).lower()
    return any(kw in error_lower for kw in keywords)


_BASE_YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "noplaylist": True,
    "skip_download": True,
    # Subtitles/translations are never used here; skipping them avoids extra
    # network round-trips during extraction.
    "extractor_args": {"youtube": {"skip": ["translated_subs"]}},
}

_MAX_VIDEO_HEIGHT = 480
_MAX_CAPTURE_ATTEMPTS = 3
_COMMON_BROWSERS = ['chrome', 'firefox', 'chromium', 'safari', 'edge']


def _extract_info(url, cookies_browser=None):
    """Run a single yt-dlp extraction and return the info dict (may raise)."""
    opts = dict(_BASE_YDL_OPTS)
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def _select_video_urls(info, max_height=_MAX_VIDEO_HEIGHT):
    """Return candidate video URLs ordered by preference (lowest height first).

    Direct (non-HLS) formats come first because they open faster in OpenCV;
    HLS (m3u8) formats are kept as a fallback for live streams that expose
    nothing else.
    """
    candidates = [
        f for f in (info or {}).get("formats", [])
        if f.get("url")
        and f.get("vcodec") != "none"
        and 0 < (f.get("height") or 0) <= max_height
    ]
    direct = [f for f in candidates if "m3u8" not in f["url"].lower()]
    hls = [f for f in candidates if "m3u8" in f["url"].lower()]
    direct.sort(key=lambda f: f.get("height") or 0)
    hls.sort(key=lambda f: f.get("height") or 0)
    return [f["url"] for f in direct + hls]


def _select_audio_url(info):
    """Return the best audio-only URL from an already extracted info dict."""
    if not info:
        return None
    audio = [
        f for f in info.get("formats", [])
        if f.get("url") and f.get("vcodec") == "none" and f.get("acodec") != "none"
    ]
    if not audio:
        return info.get("url")
    audio.sort(
        key=lambda f: (f.get("ext") == "webm", f.get("abr") or 0),
        reverse=True,
    )
    return audio[0]["url"]


def _open_capture(video_url):
    """Open a cv2.VideoCapture and verify a frame can be read."""
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap
        cap.release()
    return None


def _open_from_info(info):
    """Try opening a capture from the formats of an extracted info dict."""
    for video_url in _select_video_urls(info)[:_MAX_CAPTURE_ATTEMPTS]:
        cap = _open_capture(video_url)
        if cap is not None:
            return cap
    return None


def _try_yt_dlp_stream(url, ydl_opts):
    """Backward-compatible helper: extract with ``ydl_opts`` and open a capture.

    Exceptions from extract_info (including auth/bot errors) are propagated to
    the caller so they can be classified correctly.
    """
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)  # may raise — intentional
    return _open_from_info(info)


def _open_with_streamlink(url):
    """Fallback: resolve the stream with streamlink and open a capture."""
    try:
        import streamlink
    except ImportError:
        print("streamlink n'est pas installé, fallback ignoré.")
        return None

    try:
        print("Tentative avec streamlink...")
        streams = streamlink.streams(url)
        if not streams:
            raise ValueError("Aucun stream trouvé avec streamlink")

        stream = None
        for quality in ['480p', '360p', '240p', 'worst', 'best']:
            if quality in streams:
                stream = streams[quality]
                print(f"Stream trouvé en qualité: {quality}")
                break
        if stream is None:
            stream = list(streams.values())[0]
            print("Stream par défaut utilisé")

        stream_url = stream.url if hasattr(stream, 'url') else stream.to_url()
        cap = _open_capture(stream_url)
        if cap is not None:
            print("✓ Stream ouvert avec streamlink")
        return cap
    except Exception as e:
        print(f"Erreur avec streamlink: {e}")
        return None


def open_youtube_stream(url, cookies_browser=None):
    """Open a YouTube stream and return ``(cap, info)``.

    A single yt-dlp extraction is performed and reused both to pick the video
    format and (by the caller) to pick the audio format, instead of running
    several full extractions.  Order of attempts:

    1. yt-dlp without cookies;
    2. on bot detection, yt-dlp with browser cookies;
    3. streamlink as a last resort.

    ``info`` is the yt-dlp info dict (``None`` when streamlink was used).
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty or whitespace")

    print("Tentative avec yt-dlp...")
    last_error = None
    bot_detected = False
    t0 = time.monotonic()

    try:
        info = _extract_info(url)
        print(f"Extraction yt-dlp en {time.monotonic() - t0:.1f}s")
        cap = _open_from_info(info)
        if cap is not None:
            print("✓ Stream ouvert avec yt-dlp")
            return cap, info
    except Exception as e:
        last_error = e
        if _is_bot_detection_error(e):
            bot_detected = True
            print(f"⚠️ Détection bot YouTube: {e}")
        else:
            print(f"Erreur yt-dlp: {e}")

    if bot_detected:
        browsers_to_try = [cookies_browser] if cookies_browser else _COMMON_BROWSERS
        print("🔑 Tentative avec les cookies du navigateur...")
        for browser in browsers_to_try:
            if browser is None:
                continue
            try:
                info = _extract_info(url, cookies_browser=browser)
                cap = _open_from_info(info)
                if cap is not None:
                    print(f"✓ Stream ouvert avec yt-dlp + cookies {browser}")
                    return cap, info
            except Exception as e:
                if not _is_bot_detection_error(e):
                    print(f"Erreur avec cookies {browser}: {e}")

    cap = _open_with_streamlink(url)
    if cap is not None:
        return cap, None

    if bot_detected:
        raise ValueError(
            "YouTube a bloqué la requête (détection de bot). "
            "Connectez-vous à YouTube dans votre navigateur puis relancez. "
            "Pour forcer un navigateur, ajoutez 'youtube_cookies_browser': 'chrome' (ou 'firefox') "
            "dans setting.json."
        )

    raise ValueError(
        "Impossible d'ouvrir le stream YouTube. "
        f"Dernière erreur: {last_error}"
    )


def get_light_live_stream_url(url, cookies_browser=None):
    """Retrieves a YouTube stream and returns an opened VideoCapture.

    Thin wrapper around :func:`open_youtube_stream` kept for compatibility.
    """
    cap, _info = open_youtube_stream(url, cookies_browser=cookies_browser)
    return cap


class FactoryNode:
    node_label = 'YouTube'
    node_tag = 'YouTube'

    def __init__(self):
        pass

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        node = YoutubeNode()
        node.tag_node_name = str(node_id) + ':' + node.node_tag
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Input01Value'

        node.tag_node_input02_name = node.tag_node_name + ':' + node.TYPE_INT + ':Input02'
        node.tag_node_input02_value_name = node.tag_node_name + ':' + node.TYPE_INT + ':Input02Value'

        node.tag_node_output01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01'
        node.tag_node_output01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Output01Value'

        node.tag_node_output02_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02'
        node.tag_node_output02_value_name = node.tag_node_name + ':' + node.TYPE_TIME_MS + ':Output02Value'

        node.tag_node_button_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Button'
        node.tag_node_button_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':ButtonValue'

        node.tag_node_output_audio_name = node.tag_node_name + ':' + node.TYPE_AUDIO + ':OutputAudio'
        node.tag_node_output_audio_value_name = node.tag_node_name + ':' + node.TYPE_AUDIO + ':OutputAudioValue'

        node.tag_node_output_json_name = node.tag_node_name + ':' + node.TYPE_JSON + ':OutputJson'
        node.tag_node_output_json_value_name = node.tag_node_name + ':' + node.TYPE_JSON + ':OutputJsonValue'

        node._opencv_setting_dict = opencv_setting_dict
        node.small_window_w = node._opencv_setting_dict['input_window_width']
        node.small_window_h = node._opencv_setting_dict['input_window_height']

        black_image = np.zeros((node.small_window_h, node.small_window_w, 3), dtype=np.uint8)
        black_texture = node.convert_cv_to_dpg(
            black_image,
            node.small_window_w,
            node.small_window_h,
        )

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                node.small_window_w,
                node.small_window_h,
                black_texture,
                tag=node.tag_node_output01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )

        # Create yellow theme for buttons
        with dpg.theme() as yellow_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 153, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (255, 255, 153, 255))

        # Create blue theme for active streaming
        with dpg.theme() as blue_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 149, 237, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (65, 105, 225, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (25, 25, 112, 255))

        # Create default (inactive) theme for audio button
        with dpg.theme() as default_button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (51, 51, 55, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 66, 70, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (80, 80, 85, 255))
        
        node.yellow_button_theme = yellow_button_theme
        node.blue_button_theme = blue_button_theme
        node.default_button_theme = default_button_theme

        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=node.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_input_text(
                    tag=node.tag_node_input01_value_name,
                    label='URL',
                    width=node.small_window_w - 30,
                )

            with dpg.node_attribute(
                    tag=node.tag_node_output01_name,
                    attribute_type=dpg.mvNode_Attr_Output,
            ):
                dpg.add_image(node.tag_node_output01_value_name)

            with dpg.node_attribute(
                    tag=node.tag_node_input02_name,
                    attribute_type=dpg.mvNode_Attr_Static
            ):
                dpg.add_slider_int(
                    tag=node.tag_node_input02_value_name,
                    label="Interval(ms)",
                    width=node.small_window_w - 110,
                    default_value=33,
                    min_value=node._min_val,
                    max_value=node._max_val,
                    callback=None,
                )

            with dpg.node_attribute(
                    tag=node.tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                btn_start = dpg.add_button(
                    label=node._start_label,
                    tag=node.tag_node_button_value_name,
                    width=node.small_window_w,
                    callback=node.button,
                    user_data=node.tag_node_input01_value_name,
                )
                dpg.bind_item_theme(btn_start, yellow_button_theme)

            def add_yellow_disabled_button(label, tag):
                btnn = dpg.add_button(
                    label=label,
                    tag=tag,
                    width=node.small_window_w,
                    enabled=False,
                )
                dpg.bind_item_theme(btnn, yellow_button_theme)
                return btnn

            with dpg.node_attribute(tag=node.tag_node_output_audio_name, attribute_type=dpg.mvNode_Attr_Output):
                btn_audio = dpg.add_button(
                    label="Audio",
                    tag=node.tag_node_output_audio_value_name,
                    width=node.small_window_w,
                    callback=node._toggle_audio,
                )
                dpg.bind_item_theme(btn_audio, default_button_theme)
                
            with dpg.node_attribute(tag=node.tag_node_output_json_name, attribute_type=dpg.mvNode_Attr_Output):
                add_yellow_disabled_button("JSON", node.tag_node_output_json_value_name)

        return node


class YoutubeNode(Node):
    _ver = '0.0.1'

    node_label = 'YouTube'
    node_tag = 'YouTube'

    _opencv_setting_dict = None
    _start_label = 'Start'
    _stop_label = 'Stop'
    _loading_label = 'Loading...'

    _min_val = 1
    _max_val = 200

    def __init__(self):
        super().__init__()
        self._min_val = 1
        self._max_val = 1000
        self._start_label = "Start"
        self.node_tag = "YouTube"
        self.node_label = "YouTube"
        self.cap = None
        self.small_window_w = 240
        self.small_window_h = 135
        self.yellow_button_theme = None
        self.blue_button_theme = None
        self.default_button_theme = None
        self.is_streaming = False
        self._frame_skip_counter = 0
        # Frame counting and sync state
        self._frame_count = 0
        self._stream_start_time = None  # time.monotonic() when streaming started
        self._audio_chunk_counter = 0  # incremented for each audio chunk read

        # Audio state
        self._audio_enabled = False
        self._audio_process = None
        self._audio_thread = None
        self._audio_queue = queue.Queue(maxsize=10)
        self._audio_sr = 16000
        self._audio_chunk_duration = 5.0  # seconds per chunk
        self._audio_url = None
        self._stream_info = None
        self._audio_stop_event = threading.Event()
        self._loading = False
        self._loading_thread = None

        # Frames are read by a background thread: cap.read() blocks on the
        # network and update() runs with the DearPyGui lock held, so reading
        # there freezes the interface whenever the stream stalls.
        self._frame_interval = 0.033
        self._frame_lock = threading.Lock()
        self._pending_frame = None
        self._reader_stop = None
        self._reader_thread = None

    def convert_cv_to_dpg(self, cv_img, w, h):
        """Converts OpenCV image to DearPyGui format"""
        if cv_img is None:
            return (np.zeros(w * h * 3, dtype=np.float32)).tobytes()
        
        resized = cv2.resize(cv_img, (w, h))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        return normalized.flatten().tobytes()

    def _start_reader(self, cap):
        """Start the thread that reads frames from ``cap``."""
        stop_event = threading.Event()
        self._reader_stop = stop_event
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(cap, stop_event),
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, cap, stop_event):
        """Background thread: reads one frame per interval, keeps the latest.

        The thread owns ``cap`` and is the only one to release it: releasing
        a capture from another thread while read() is in progress crashes
        OpenCV.
        """
        failures = 0
        try:
            while not stop_event.is_set():
                started = time.monotonic()
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    print(f"❌ Erreur de lecture: {e}")
                    ret, frame = False, None
                if stop_event.is_set():
                    break

                if ret and frame is not None:
                    with self._frame_lock:
                        self._pending_frame = frame
                    failures = 0
                    min_delay = 0.001
                else:
                    failures += 1
                    if failures > 150:
                        print("⚠️ Avertissement: Stream pourrait être terminé ou avoir un problème")
                        failures = 0
                    min_delay = 0.01

                elapsed = time.monotonic() - started
                stop_event.wait(max(min_delay, self._frame_interval - elapsed))
        finally:
            cap.release()

    def _release_capture(self):
        """Stop reading and release the capture, without ever blocking."""
        stop_event = self._reader_stop
        cap = self.cap
        self._reader_stop = None
        self._reader_thread = None
        self.cap = None
        if stop_event is not None:
            # The reader thread releases the capture once read() returns.
            stop_event.set()
        elif cap is not None:
            cap.release()
        with self._frame_lock:
            self._pending_frame = None

    def _toggle_audio(self, sender, data, user_data=None):
        """Toggle audio extraction on/off."""
        self._audio_enabled = not self._audio_enabled
        if self._audio_enabled:
            dpg.bind_item_theme(sender, self.yellow_button_theme)
            # Start audio capture if currently streaming
            if self.is_streaming and self._audio_url:
                self._start_audio_capture()
        else:
            dpg.bind_item_theme(sender, self.default_button_theme)
            self._stop_audio_capture()

    def _get_audio_stream_url(self, youtube_url):
        """Get the best audio-only stream URL from YouTube.

        Reuses the info dict obtained when the video stream was opened to
        avoid a second full yt-dlp extraction; only falls back to a fresh
        extraction when that info is unavailable (e.g. streamlink fallback).
        """
        audio_url = _select_audio_url(self._stream_info)
        if audio_url:
            return audio_url

        cookies_browser = (self._opencv_setting_dict or {}).get("youtube_cookies_browser")

        def _extract(ydl_opts):
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info.get("url")

        base_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestaudio[ext=webm]/bestaudio/best",
            "nocheckcertificate": True,
        }

        try:
            return _extract(base_opts)
        except Exception as e:
            if not _is_bot_detection_error(e):
                print(f"❌ Erreur extraction URL audio: {e}")
                return None

        # Retry with browser cookies on bot detection
        browsers_to_try = [cookies_browser] if cookies_browser else ['chrome', 'firefox', 'chromium', 'safari', 'edge']
        print("🔑 Tentative audio avec les cookies du navigateur...")
        for browser in browsers_to_try:
            if browser is None:
                continue
            try:
                opts = dict(base_opts)
                opts["cookiesfrombrowser"] = (browser,)
                url = _extract(opts)
                if url:
                    print(f"✓ URL audio extraite avec cookies {browser}")
                    return url
            except Exception as e:
                if not _is_bot_detection_error(e):
                    print(f"❌ Erreur extraction URL audio ({browser}): {e}")

        print("❌ Impossible d'extraire l'URL audio: YouTube bloque la requête. Connectez-vous à YouTube dans votre navigateur.")
        return None

    def _start_audio_capture(self):
        """Start the ffmpeg subprocess to capture audio from the stream."""
        self._stop_audio_capture()
        
        if not self._audio_url:
            return

        ffmpeg_exe = _get_ffmpeg_exe()
        if ffmpeg_exe is None:
            print("❌ ffmpeg non trouvé pour l'extraction audio")
            return

        self._audio_stop_event.clear()
        self._audio_thread = threading.Thread(
            target=self._audio_reader_loop,
            args=(ffmpeg_exe, self._audio_url),
            daemon=True,
        )
        self._audio_thread.start()

    def _audio_reader_loop(self, ffmpeg_exe, audio_url):
        """Background thread: reads PCM audio from ffmpeg stdout."""
        try:
            cmd = [
                ffmpeg_exe,
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
                "-i", audio_url,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self._audio_sr),
                "-ac", "1",
                "-f", "s16le",
                "-",
            ]
            self._audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            bytes_per_chunk = int(self._audio_sr * self._audio_chunk_duration) * 2  # 16-bit mono

            while not self._audio_stop_event.is_set():
                raw = self._audio_process.stdout.read(bytes_per_chunk)
                if not raw:
                    break
                audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                chunk_idx = self._audio_chunk_counter
                self._audio_chunk_counter += 1
                audio_dict = {
                    'data': audio_np,
                    'sample_rate': self._audio_sr,
                    'channels': 1,
                    'chunk_index': chunk_idx,
                    'step_duration': self._audio_chunk_duration,
                }
                # Non-blocking put: discard old chunks if queue is full
                try:
                    self._audio_queue.put_nowait(audio_dict)
                except queue.Full:
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._audio_queue.put_nowait(audio_dict)

        except Exception as e:
            if not self._audio_stop_event.is_set():
                print(f"❌ Erreur audio reader: {e}")
        finally:
            if self._audio_process and self._audio_process.poll() is None:
                self._audio_process.kill()
                self._audio_process = None

    def _stop_audio_capture(self):
        """Stop the audio capture thread and ffmpeg process."""
        self._audio_stop_event.set()
        if self._audio_process and self._audio_process.poll() is None:
            self._audio_process.kill()
            self._audio_process = None
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)
        self._audio_thread = None
        # Drain the queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def button(self, sender, data, user_data):
        tag_parts = user_data.split(':')
        tag_node_name = ':'.join(tag_parts[:2])
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        label = dpg.get_item_label(tag_node_button_value_name)
        youtube_url = dpg.get_value(user_data)

        if label == self._start_label:
            if not youtube_url or not isinstance(youtube_url, str) or not youtube_url.strip():
                print("❌ Erreur: Veuillez entrer une URL YouTube valide")
                return

            if self._loading:
                return

            self._loading = True
            self._release_capture()

            print(f"🔄 Ouverture du stream YouTube: {youtube_url}")
            dpg.set_item_label(tag_node_button_value_name, self._loading_label)

            def _open_stream():
                try:
                    cap, info = open_youtube_stream(
                        youtube_url,
                        cookies_browser=(self._opencv_setting_dict or {}).get("youtube_cookies_browser"),
                    )
                    self._stream_info = info

                    if cap is None or not cap.isOpened():
                        print("❌ Erreur: Impossible d'ouvrir le stream")
                        if dpg.does_item_exist(tag_node_button_value_name):
                            dpg.set_item_label(tag_node_button_value_name, self._start_label)
                        return

                    self.cap = cap
                    print("✅ Stream YouTube démarré avec succès!")
                    self._frame_skip_counter = 0
                    self._frame_count = 0
                    self._audio_chunk_counter = 0
                    self._stream_start_time = time.monotonic()
                    self.is_streaming = True
                    self._start_reader(cap)

                    if dpg.does_item_exist(tag_node_button_value_name):
                        dpg.set_item_label(tag_node_button_value_name, self._stop_label)
                        if self.blue_button_theme is not None:
                            dpg.bind_item_theme(tag_node_button_value_name, self.blue_button_theme)

                    self._audio_url = self._get_audio_stream_url(youtube_url)
                    if self._audio_enabled and self._audio_url:
                        self._start_audio_capture()

                except ValueError as e:
                    print(f"❌ Erreur: {e}")
                    self._release_capture()
                    self.is_streaming = False
                    if dpg.does_item_exist(tag_node_button_value_name):
                        dpg.set_item_label(tag_node_button_value_name, self._start_label)
                except Exception as e:
                    print(f"❌ Erreur inattendue: {e}")
                    self._release_capture()
                    self.is_streaming = False
                    if dpg.does_item_exist(tag_node_button_value_name):
                        dpg.set_item_label(tag_node_button_value_name, self._start_label)
                finally:
                    self._loading = False

            self._loading_thread = threading.Thread(target=_open_stream, daemon=True)
            self._loading_thread.start()

        elif label == self._stop_label:
            if self.cap is not None:
                print("⏹️ Stream YouTube arrêté")
            self._release_capture()

            self.is_streaming = False
            self._stop_audio_capture()
            self._audio_url = None
            self._stream_info = None
            dpg.set_item_label(tag_node_button_value_name, self._start_label)

            if self.yellow_button_theme is not None:
                dpg.bind_item_theme(tag_node_button_value_name, self.yellow_button_theme)

    def update(self, node_id, connection_list, node_image_dict, node_result_dict, node_audio_dict):
        """Updates the video stream image."""
        tag_node_name = f"{node_id}:{self.node_tag}"
        output_value01_tag = f"{tag_node_name}:{self.TYPE_IMAGE}:Output01Value"

        self.current_time = time.time()

        if not hasattr(self, "_last_frame_time"):
            self._last_frame_time = 0
        
        try:
            slider_tag = f"{tag_node_name}:{self.TYPE_INT}:Input02Value"
            self._frame_interval = max(1, dpg_get_value(slider_tag)) / 1000
        except (ValueError, KeyError, AttributeError, TypeError):
            self._frame_interval = 0.033

        with self._frame_lock:
            frame = self._pending_frame
            self._pending_frame = None

        if frame is not None and self.is_streaming:
            self._last_frame = frame
            self._frame_count += 1
            self._last_frame_time = self.current_time
            try:
                texture = self.convert_cv_to_dpg(frame, self.small_window_w, self.small_window_h)
                dpg_set_value(output_value01_tag, texture)
            except Exception as e:
                print(f"❌ Erreur d'affichage: {e}")

        # Get audio chunk if audio is enabled
        audio_chunk_data = None
        if self._audio_enabled:
            try:
                audio_chunk_data = self._audio_queue.get_nowait()
            except queue.Empty:
                pass

        # Calculate timestamp based on frame count and interval
        # Use frame_interval as the effective "1/fps" value
        current_frame = self._frame_count
        effective_fps = 1.0 / self._frame_interval if self._frame_interval > 0 else 30.0
        frame_timestamp = current_frame / effective_fps if current_frame > 0 else None
        pts_ms = frame_timestamp * 1000.0 if frame_timestamp is not None else None

        # Inject pts_ms into audio dict for A/V sync alignment
        if audio_chunk_data is not None and pts_ms is not None:
            audio_chunk_data = dict(audio_chunk_data)
            audio_chunk_data["pts_ms"] = pts_ms

        # Build FramePacket JSON metadata (same as VideoNode)
        json_output = None
        current_image = getattr(self, "_last_frame", None)
        if current_image is not None and pts_ms is not None:
            audio_chunk_index = 0
            if isinstance(audio_chunk_data, dict):
                audio_chunk_index = audio_chunk_data.get("chunk_index", 0)
            now = time.monotonic()
            fp = FramePacket(
                frame_index=current_frame,
                pts_ms=pts_ms,
                audio_chunk_index=audio_chunk_index,
                image=current_image,
                audio_data=audio_chunk_data,
                pipeline_entry_ts=now,
                pipeline_exit_ts=now,
            )
            json_output = fp.to_metadata()

        return {
            "image": current_image,
            "json": json_output,
            "audio": audio_chunk_data,
            "timestamp": frame_timestamp,
        }
    
    def close(self, node_id):
        """Clean up resources when node is closed."""
        if self._loading_thread and self._loading_thread.is_alive():
            self._loading = False
            self._loading_thread.join(timeout=2.0)
        self._stop_audio_capture()
        self._release_capture()
        self.is_streaming = False

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'

        pos = dpg.get_item_pos(tag_node_name)
        youtube_url = dpg_get_value(tag_node_input01_value_name)
        interval_time = dpg_get_value(tag_node_input02_value_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_input01_value_name] = youtube_url
        setting_dict[tag_node_input02_value_name] = interval_time

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        tag_node_input01_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':Input01Value'
        tag_node_input02_value_name = tag_node_name + ':' + self.TYPE_INT + ':Input02Value'

        youtube_url = setting_dict[tag_node_input01_value_name]
        interval_time = setting_dict[tag_node_input02_value_name]

        dpg_set_value(tag_node_input01_value_name, youtube_url)
        dpg_set_value(tag_node_input02_value_name, interval_time)
