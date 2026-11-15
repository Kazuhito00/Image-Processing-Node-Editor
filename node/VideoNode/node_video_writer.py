#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import datetime
import json
import subprocess
import tempfile
import threading
import time
import logging

import shutil

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from node_editor.util import dpg_get_value, dpg_set_value

from node.node_abc import DpgNodeABC
#from node_editor.util import convert_cv_to_dpg
from node.basenode import Node
from node.VideoNode.sync import FramePacket, SyncVideoWriter
from node.VideoNode.av_encoder import PyAVEncoder, _AV_AVAILABLE

logger = logging.getLogger(__name__)

try:
    import ffmpeg
    _FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    ffmpeg = None  # type: ignore[assignment]
    _FFMPEG_PYTHON_AVAILABLE = False

try:
    import imageio_ffmpeg
    _IMAGEIO_FFMPEG_AVAILABLE = True
except ImportError:
    imageio_ffmpeg = None  # type: ignore[assignment]
    _IMAGEIO_FFMPEG_AVAILABLE = False


def _get_ffmpeg_exe() -> str:
    """Return a usable ffmpeg executable path.

    Resolution order:
    1. imageio-ffmpeg bundled binary (no system install required).
       Falls through if imageio_ffmpeg is not installed or raises RuntimeError.
    2. ffmpeg binary found on the system PATH via shutil.which.
       Falls through if shutil.which returns None.
    3. Plain ``'ffmpeg'`` string as last resort, letting the OS raise an
       informative error if the binary is truly absent.
    """
    if _IMAGEIO_FFMPEG_AVAILABLE:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except RuntimeError:
            pass
    found = shutil.which("ffmpeg")
    return found if found is not None else "ffmpeg"

try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None  # type: ignore[assignment]
    _SOUNDFILE_AVAILABLE = False

FFMPEG_AVAILABLE = _FFMPEG_PYTHON_AVAILABLE and _SOUNDFILE_AVAILABLE

if not FFMPEG_AVAILABLE:
    _missing_pkgs = []
    if not _FFMPEG_PYTHON_AVAILABLE:
        _missing_pkgs.append("ffmpeg-python")
    if not _SOUNDFILE_AVAILABLE:
        _missing_pkgs.append("soundfile")
    import warnings
    warnings.warn(
        "{} not installed. VideoWriter will save video WITHOUT audio. "
        "Fix: pip install {}".format(
            " and ".join(_missing_pkgs), " ".join(_missing_pkgs)
        ),
        RuntimeWarning,
        stacklevel=1,
    )
    del _missing_pkgs

def slow_motion_interpolation(prev_frame, next_frame, alpha):
    """ Generates smooth intermediate frame between 2 images """
    return cv2.addWeighted(prev_frame, 1 - alpha, next_frame, alpha, 0)


# VideoWriter write-mode constants
WRITE_MODE_STANDARD = "Standard"
WRITE_MODE_ON_DISK   = "On-disk"
WRITE_MODES = [WRITE_MODE_STANDARD, WRITE_MODE_ON_DISK]



class FactoryNode:
    node_label = 'VideoWriter'
    node_tag = 'VideoWriter'
    

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

        node = VideoWriterNode()
        node.tag_node_name = str(node_id) + ':' + node.node_tag
        node.tag_node_input01_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01'
        node.tag_node_input01_value_name = node.tag_node_name + ':' + node.TYPE_IMAGE + ':Input01Value'

        node.tag_node_button_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Button'
        node.tag_node_button_value_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':ButtonValue'
        node.tag_node_progress_name = node.tag_node_name + ':' + node.TYPE_TEXT + ':Progress'


        node._opencv_setting_dict = opencv_setting_dict
        small_window_w = node._opencv_setting_dict['process_width']
        small_window_h = node._opencv_setting_dict['process_height']


        black_image = np.zeros((small_window_w, small_window_h, 3))
        black_texture = node.convert_cv_to_dpg(
            black_image,
            small_window_w,
            small_window_h,
        )

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                small_window_w,
                small_window_h,
                black_texture,
                tag=node.tag_node_input01_value_name,
                format=dpg.mvFormat_Float_rgb,
            )


        with dpg.node(
                tag=node.tag_node_name,
                parent=parent,
                label=self.node_label,
                pos=pos,
        ):

            with dpg.node_attribute(
                    tag=node.tag_node_input01_name,
                    attribute_type=dpg.mvNode_Attr_Input,
            ):
                dpg.add_image(node.tag_node_input01_value_name)

            # Add format selector
            with dpg.node_attribute(
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    tag=node.tag_node_name + ':Format',
                    items=['MP4', 'AVI', 'MKV'],
                    default_value='MP4',
                    width=small_window_w,
                )

            # Write-mode selector: Standard (video+audio in-memory merge)
            # or On-disk (direct H.264 write, no audio, constant RAM usage).
            with dpg.node_attribute(
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_combo(
                    tag=node.tag_node_name + ':WriteMode',
                    items=WRITE_MODES,
                    label="Mode",
                    default_value=WRITE_MODE_STANDARD,
                    width=small_window_w,
                )

            with dpg.node_attribute(
                    tag=node.tag_node_button_name,
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_button(
                    label=node._start_label,
                    tag=node.tag_node_button_value_name,
                    width=small_window_w,
                    callback=node._recording_button,
                    user_data=node.tag_node_name,
                )
            
            # Add progress bar for merge operation
            with dpg.node_attribute(
                    attribute_type=dpg.mvNode_Attr_Static,
            ):
                dpg.add_progress_bar(
                    label="Merge Progress",
                    tag=node.tag_node_progress_name,
                    default_value=0.0,
                    overlay="",
                    width=small_window_w,
                    show=False,  # Hidden by default
                )

        return node



class VideoWriterNode(Node):
    _ver = '0.0.2'

    node_label = 'VideoWriter'
    node_tag = 'VideoWriter'

    _opencv_setting_dict = None

    _video_writer_dict = {}
    _mkv_metadata_dict = {}  # Store audio and JSON metadata for MKV files
    _mkv_file_handles = {}  # Store file handles for MKV metadata tracks
    _audio_samples_dict = {}  # Store audio samples during recording for merging
    _last_chunk_index_dict = {}  # Track last appended chunk_index per node to avoid duplicates
    _recording_metadata_dict = {}  # Store metadata about ongoing recordings
    _merge_threads_dict = {}  # Store merge threads for async operations
    _merge_progress_dict = {}  # Store merge progress (0.0 to 1.0)
    _sync_writers_dict = {}  # SyncVideoWriter instances keyed by tag_node_name
    _frame_counter_dict = {}  # Per-node frame counter for FramePacket construction
    _ondisk_writers_dict = {}  # PyAVEncoder instances for on-disk mode (keyed by tag_node_name)
    _start_label = 'Start'
    _stop_label = 'Stop'
    
    # Constants for file wait logic
    # These control the behavior when waiting for the video file to be written to disk
    # before starting the audio/video merge operation
    _FILE_WAIT_TIMEOUT = 5.0  # Maximum seconds to wait for video file (range: 1.0-10.0)
    _FILE_WAIT_INTERVAL = 0.1  # Check interval in seconds (range: 0.05-0.5)
    _FILE_FLUSH_DELAY = 0.1  # Additional delay after file exists to ensure flush (range: 0.05-0.5)

    _prev_frame_flag = False

    def __init__(self):
        pass



    def update(
        self,
        node_id,
        connection_list,
        node_image_dict,
        node_result_dict,
        node_audio_dict,
    ):
        tag_node_name = str(node_id) + ':' + self.node_tag
        input_value01_tag = tag_node_name + ':' + self.TYPE_IMAGE + ':Input01Value'
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'
        tag_node_progress_name = tag_node_name + ':' + self.TYPE_TEXT + ':Progress'

        # Update merge progress bar if merge is in progress
        if tag_node_name in self._merge_progress_dict:
            progress = self._merge_progress_dict[tag_node_name]
            if dpg.does_item_exist(tag_node_progress_name):
                dpg.configure_item(tag_node_progress_name, show=True)
                dpg.set_value(tag_node_progress_name, progress)
                dpg.configure_item(tag_node_progress_name, overlay=f"Merging: {int(progress * 100)}%")
            
            # Check if merge thread has completed
            if tag_node_name in self._merge_threads_dict:
                thread = self._merge_threads_dict[tag_node_name]
                if not thread.is_alive():
                    # Thread completed, clean up
                    self._merge_threads_dict.pop(tag_node_name)
                    self._merge_progress_dict.pop(tag_node_name)
                    if dpg.does_item_exist(tag_node_progress_name):
                        dpg.configure_item(tag_node_progress_name, show=False)
                        dpg.set_value(tag_node_progress_name, 0.0)
                        dpg.configure_item(tag_node_progress_name, overlay="")

        connection_info_src = ''
        for connection_info in connection_list:
            connection_info_src = connection_info[0]
            connection_info_src = connection_info_src.split(':')[:2]
            connection_info_src = ':'.join(connection_info_src)

        small_window_w = self._opencv_setting_dict['process_width']
        small_window_h = self._opencv_setting_dict['process_height']
        writer_width = self._opencv_setting_dict['video_writer_width']
        writer_height = self._opencv_setting_dict['video_writer_height']

        frame = node_image_dict.get(connection_info_src, None)
        
        # Get audio and JSON data if available
        audio_data = node_audio_dict.get(connection_info_src, None)
        json_data = node_result_dict.get(connection_info_src, None)


        if frame is not None:
            rec_frame = copy.deepcopy(frame)

            # --- On-disk mode: write directly via PyAVEncoder (or cv2 fallback) ---
            if tag_node_name in self._ondisk_writers_dict:
                writer_frame = cv2.resize(rec_frame,
                                          (writer_width, writer_height),
                                          interpolation=cv2.INTER_CUBIC)
                od_writer = self._ondisk_writers_dict[tag_node_name]
                frame_idx = self._frame_counter_dict.get(tag_node_name, 0)
                self._frame_counter_dict[tag_node_name] = frame_idx + 1
                meta = self._recording_metadata_dict.get(tag_node_name, {})
                fps_val = meta.get("fps", 30.0)
                pts_ms = (frame_idx / max(fps_val, 1)) * 1000.0
                if isinstance(od_writer, cv2.VideoWriter):
                    od_writer.write(writer_frame)
                else:
                    try:
                        od_writer.write_video_frame(writer_frame, pts_ms)
                    except Exception as e:
                        logger.error("PyAVEncoder write_video_frame error: %s", e)

            # --- Standard mode: cv2.VideoWriter + audio buffering ---
            elif tag_node_name in self._video_writer_dict:

                writer_frame = cv2.resize(rec_frame,
                                          (writer_width, writer_height),
                                          interpolation=cv2.INTER_CUBIC)

                # Build a FramePacket from upstream JSON metadata (if available)
                # or fall back to a counter-based packet.
                frame_idx = self._frame_counter_dict.get(tag_node_name, 0)
                self._frame_counter_dict[tag_node_name] = frame_idx + 1

                if isinstance(json_data, dict) and "_frame_packet" in json_data:
                    packet = FramePacket.from_metadata(
                        json_data, writer_frame, audio_data
                    )
                else:
                    # Fallback: build from frame counter using fps stored in recording metadata
                    meta = self._recording_metadata_dict.get(tag_node_name, {})
                    fps_val = meta.get("fps", 30.0)
                    now = time.monotonic()
                    packet = FramePacket(
                        frame_index=frame_idx,
                        pts_ms=(frame_idx / max(fps_val, 1)) * 1000.0,
                        audio_chunk_index=0,
                        image=writer_frame,
                        audio_data=audio_data,
                        pipeline_entry_ts=now,
                        pipeline_exit_ts=now,
                    )

                # Enqueue into the priority-queue based sync writer
                if tag_node_name in self._sync_writers_dict:
                    sync_writer = self._sync_writers_dict[tag_node_name]
                    sync_writer.enqueue(packet)
                    cv2_writer = self._video_writer_dict[tag_node_name]
                    sync_writer.flush_ready(
                        lambda img, _pts: cv2_writer.write(img)
                    )
                else:
                    # No sync writer yet (race at startup) → write directly
                    self._video_writer_dict[tag_node_name].write(writer_frame)
                
                # Collect audio samples for final merge (for all formats)
                if audio_data is not None and tag_node_name in self._audio_samples_dict:
                    # audio_data can be a dict (from concat node with multiple slots) or a single chunk
                    if isinstance(audio_data, dict):
                        # Check if this is a multi-slot concat output or single audio chunk from video node
                        # Multi-slot: {0: audio_chunk, 1: audio_chunk, ...}
                        # Single chunk: {'data': array, 'sample_rate': int}
                        
                        if 'data' in audio_data and 'sample_rate' in audio_data:
                            # Single audio chunk from video node — deduplicate by chunk_index.
                            # Audio chunks are sliding-window (chunk_duration > step_duration),
                            # so only keep the non-overlapping step portion to avoid concatenating
                            # audio that is (chunk_duration / step_duration)× too long, which
                            # would cause a progressive A/V drift proportional to that ratio.
                            incoming_idx = audio_data.get('chunk_index', None)
                            last_idx = self._last_chunk_index_dict.get(tag_node_name, -1)
                            if incoming_idx is None or incoming_idx != last_idx:
                                self._last_chunk_index_dict[tag_node_name] = incoming_idx
                                chunk_data = audio_data['data']
                                sr = audio_data['sample_rate']
                                step_dur = audio_data.get('step_duration', None)
                                if step_dur is not None and sr > 0:
                                    # Trim to the non-overlapping step portion only
                                    step_samples = int(step_dur * sr)
                                    chunk_data = chunk_data[:step_samples]
                                # Trim leading audio from the first collected chunk so
                                # that the audio track starts at the same time as the
                                # first video frame.  Without this correction the audio
                                # is ~(1/fps) seconds ahead of the images because
                                # VideoNode uses a 1-based frame counter: frame 1 maps
                                # to pts_ms ≈ 33 ms while the audio chunk for that frame
                                # starts at chunk_index * step_dur = 0 ms.  The image
                                # therefore appears ahead of the audio during playback.
                                n_current = len(self._audio_samples_dict[tag_node_name])
                                if n_current == 0 and step_dur is not None and sr > 0:
                                    chunk_idx_val = incoming_idx if incoming_idx is not None else 0
                                    chunk_start_s = chunk_idx_val * step_dur
                                    lead_s = max(0.0, packet.pts_ms / 1000.0 - chunk_start_s)
                                    lead_samples = int(lead_s * sr)
                                    if lead_samples > 0:
                                        chunk_data = chunk_data[lead_samples:]
                                        logger.debug(
                                            "VideoWriter[%s]: Trimmed %d leading samples "
                                            "(%.1f ms) from first audio chunk to align with "
                                            "first video frame (frame pts_ms=%.1f ms, "
                                            "chunk_start=%.1f ms).",
                                            tag_node_name,
                                            lead_samples,
                                            lead_s * 1000.0,
                                            packet.pts_ms,
                                            chunk_start_s * 1000.0,
                                        )
                                self._audio_samples_dict[tag_node_name].append(chunk_data)
                                n_collected = len(self._audio_samples_dict[tag_node_name])
                                if n_collected == 1:
                                    logger.info(
                                        "VideoWriter[%s]: First audio chunk received "
                                        "(chunk_index=%s, samples=%s, SR=%s Hz, step_dur=%s). Audio recording active.",
                                        tag_node_name,
                                        incoming_idx,
                                        len(chunk_data) if hasattr(chunk_data, '__len__') else 'n/a',
                                        sr,
                                        step_dur,
                                    )
                                else:
                                    logger.debug(
                                        "VideoWriter[%s]: audio chunk index=%s samples=%s SR=%s total_chunks=%d",
                                        tag_node_name,
                                        incoming_idx,
                                        len(chunk_data) if hasattr(chunk_data, '__len__') else 'n/a',
                                        sr,
                                        n_collected,
                                    )
                                if tag_node_name in self._recording_metadata_dict:
                                    self._recording_metadata_dict[tag_node_name]['sample_rate'] = sr
                        else:
                            # Concat node output: {slot_idx: audio_chunk}
                            self._collect_concat_audio(tag_node_name, audio_data, packet)
                    else:
                        # Single audio chunk as numpy array
                        if isinstance(audio_data, np.ndarray):
                            self._audio_samples_dict[tag_node_name].append(audio_data)
                            n_collected = len(self._audio_samples_dict[tag_node_name])
                            if n_collected == 1:
                                logger.info(
                                    "VideoWriter[%s]: First raw ndarray audio received (samples=%d). Audio recording active.",
                                    tag_node_name, len(audio_data),
                                )
                            else:
                                logger.debug(
                                    "VideoWriter[%s]: ndarray audio samples=%d total_chunks=%d",
                                    tag_node_name, len(audio_data), n_collected,
                                )
                elif tag_node_name in self._video_writer_dict and tag_node_name in self._audio_samples_dict:
                    # Recording is active but no audio arriving — warn once every 60 frames
                    no_audio_count = self._frame_counter_dict.get(tag_node_name, 0)
                    if no_audio_count == 1:
                        logger.warning(
                            "VideoWriter[%s]: Recording active but audio_data=None on first frame. "
                            "Source: %s. Check that the Video node 'Frames only' checkbox is unchecked "
                            "and that audio preprocessing has completed.",
                            tag_node_name, connection_info_src,
                        )
                    elif no_audio_count % 300 == 0:
                        n_collected = len(self._audio_samples_dict.get(tag_node_name, []))
                        logger.warning(
                            "VideoWriter[%s]: %d frames recorded, %d audio chunks collected (audio_data=None this frame).",
                            tag_node_name, no_audio_count, n_collected,
                        )
                
                # Write audio and JSON data to MKV metadata tracks if applicable
                if tag_node_name in self._mkv_metadata_dict:
                    metadata = self._mkv_metadata_dict[tag_node_name]
                    file_base = metadata['file_path'].rsplit('.', 1)[0]
                    metadata_dir = file_base + '_metadata'
                    
                    # Write audio chunks if available
                    if audio_data is not None:
                        for slot_idx, audio_chunk in (audio_data.items() if isinstance(audio_data, dict) else enumerate([audio_data])):
                            # Create audio track file if not exists
                            if slot_idx not in metadata['audio_handles']:
                                audio_file = os.path.join(metadata_dir, f'audio_slot_{slot_idx}.jsonl')
                                metadata['audio_handles'][slot_idx] = open(audio_file, 'a')
                            
                            handle = metadata['audio_handles'][slot_idx]
                            # Store audio chunk as JSON (will be written to file)
                            handle.write(json.dumps({'slot': slot_idx, 'data': audio_chunk.tolist() if hasattr(audio_chunk, 'tolist') else str(audio_chunk)}) + '\n')
                            handle.flush()  # Ensure data is written
                    
                    # Write JSON data if available
                    if json_data is not None:
                        for slot_idx, json_chunk in (json_data.items() if isinstance(json_data, dict) else enumerate([json_data])):
                            # Create JSON track file if not exists
                            if slot_idx not in metadata['json_handles']:
                                json_file = os.path.join(metadata_dir, f'json_slot_{slot_idx}.jsonl')
                                metadata['json_handles'][slot_idx] = open(json_file, 'a')
                            
                            handle = metadata['json_handles'][slot_idx]
                            # Write JSON chunk
                            handle.write(json.dumps({'slot': slot_idx, 'data': json_chunk}) + '\n')
                            handle.flush()  # Ensure data is written


                rec_frame = cv2.circle(rec_frame, (10, 10),
                                       50, (0, 0, 255),
                                       thickness=-1)

            texture = self.convert_cv_to_dpg(
                rec_frame,
                small_window_w,
                small_window_h,
            )
            dpg_set_value(input_value01_tag, texture)
        else:
            label = dpg.get_item_label(tag_node_button_value_name)
            if label == self._stop_label and self._prev_frame_flag:

                self._recording_button(None, None, tag_node_name)

                black_image = np.zeros((small_window_w, small_window_h, 3))

                texture = self.convert_cv_to_dpg(
                    black_image,
                    small_window_w,
                    small_window_h,
                )
                dpg_set_value(input_value01_tag, texture)

        if frame is not None:
            self._prev_frame_flag = True
        else:
            self._prev_frame_flag = False

        return {"image":frame, "json":None, "audio":None}

    def _collect_concat_audio(self, tag_node_name, audio_data, packet):
        """Collect one frame's worth of ImageConcat audio into _audio_samples_dict.

        Applies the same chunk_index deduplication and step_duration/leading-audio
        trim as the single-chunk (input:video) path so that sliding-window audio
        chunks are handled identically regardless of whether they originate from
        a VideoNode or pass through an ImageConcat AUDIO slot.

        Parameters
        ----------
        tag_node_name : str
            Key used for all per-node dicts (e.g. "0:VideoWriter").
        audio_data : dict
            Concat-style audio dict: {slot_idx: audio_chunk_dict_or_ndarray}.
        packet : FramePacket
            The current FramePacket (used for pts_ms-based leading-trim).
        """
        if tag_node_name not in self._audio_samples_dict:
            return

        if not audio_data:
            return

        first_slot = min(audio_data.keys())
        first_chunk = audio_data[first_slot]
        incoming_idx = (
            first_chunk.get('chunk_index', None)
            if isinstance(first_chunk, dict)
            else None
        )
        last_idx = self._last_chunk_index_dict.get(tag_node_name, -1)
        if incoming_idx is not None and incoming_idx == last_idx:
            # Duplicate chunk — skip (same sliding-window step already collected)
            return

        if incoming_idx is not None:
            self._last_chunk_index_dict[tag_node_name] = incoming_idx

        audio_chunks = []
        sample_rate = None
        step_dur = None

        for slot_idx in sorted(audio_data.keys()):
            audio_chunk = audio_data[slot_idx]
            # Handle dict format: {'data': array, 'sample_rate': int, ...}
            if isinstance(audio_chunk, dict) and 'data' in audio_chunk:
                chunk_data = audio_chunk['data']
                sr = audio_chunk.get('sample_rate', None)
                if step_dur is None:
                    step_dur = audio_chunk.get('step_duration', None)
                if step_dur is not None and sr and sr > 0:
                    step_samples = int(step_dur * sr)
                    chunk_data = chunk_data[:step_samples]
                audio_chunks.append(chunk_data)
                if sample_rate is None and sr is not None:
                    sample_rate = sr
            elif isinstance(audio_chunk, np.ndarray):
                audio_chunks.append(audio_chunk)

        if not audio_chunks:
            return

        merged_chunk = np.concatenate(audio_chunks)

        # Trim leading audio from the first collected chunk so that the audio
        # track starts at the same time as the first video frame (mirrors the
        # single-chunk path).
        #
        # Prefer pts_ms stored inside the audio-chunk dict (set by VideoNode and
        # preserved verbatim through ImageConcat AUDIO slots) over packet.pts_ms.
        # When ImageConcat is in the pipeline and _frame_packet is not forwarded
        # by an older ImageConcat version, VideoWriter uses a counter-based
        # fallback packet whose pts_ms is 0 — causing the leading trim to be
        # miscalculated by up to step_duration seconds and producing A/V desync.
        chunk_pts_ms = (
            first_chunk.get('pts_ms', None)
            if isinstance(first_chunk, dict)
            else None
        )
        effective_pts_ms = chunk_pts_ms if chunk_pts_ms is not None else packet.pts_ms

        n_current = len(self._audio_samples_dict[tag_node_name])
        if n_current == 0 and step_dur is not None and sample_rate and sample_rate > 0:
            chunk_idx_val = incoming_idx if incoming_idx is not None else 0
            chunk_start_s = chunk_idx_val * step_dur
            lead_s = max(0.0, effective_pts_ms / 1000.0 - chunk_start_s)
            lead_samples = int(lead_s * sample_rate)
            if lead_samples > 0:
                merged_chunk = merged_chunk[lead_samples:]
                logger.debug(
                    "VideoWriter[%s]: Trimmed %d leading samples "
                    "(%.1f ms) from first ImageConcat audio chunk "
                    "(frame pts_ms=%.1f ms, chunk_start=%.1f ms).",
                    tag_node_name,
                    lead_samples,
                    lead_s * 1000.0,
                    effective_pts_ms,
                    chunk_start_s * 1000.0,
                )

        self._audio_samples_dict[tag_node_name].append(merged_chunk)
        n_collected = len(self._audio_samples_dict[tag_node_name])
        if n_collected == 1:
            logger.info(
                "VideoWriter[%s]: First ImageConcat audio received "
                "(chunk_index=%s, slots=%d, merged_samples=%d, SR=%s Hz). "
                "Audio recording active.",
                tag_node_name, incoming_idx,
                len(audio_chunks), len(merged_chunk), sample_rate,
            )
        else:
            logger.debug(
                "VideoWriter[%s]: ImageConcat audio chunk_index=%s "
                "slots=%d merged_samples=%d total_chunks=%d",
                tag_node_name, incoming_idx,
                len(audio_chunks), len(merged_chunk), n_collected,
            )

        # Update sample rate if found
        if sample_rate is not None and tag_node_name in self._recording_metadata_dict:
            self._recording_metadata_dict[tag_node_name]['sample_rate'] = sample_rate

    def _close_metadata_handles(self, metadata):
        """Helper method to close all metadata file handles"""
        # Close all audio handles
        for handle in metadata.get('audio_handles', {}).values():
            try:
                if not handle.closed:
                    handle.close()
            except Exception as e:
                logger.warning("Error closing audio handle: %s", e)
        
        # Close all JSON handles
        for handle in metadata.get('json_handles', {}).values():
            try:
                if not handle.closed:
                    handle.close()
            except Exception as e:
                logger.warning("Error closing JSON handle: %s", e)

    def _merge_audio_video_ffmpeg(self, video_path, audio_samples, sample_rate, output_path, progress_callback=None):
        """
        Merge video and audio using ffmpeg.
        
        Args:
            video_path: Path to the temporary video file (no audio)
            audio_samples: List of numpy arrays containing audio samples
            sample_rate: Audio sample rate (e.g., 22050, 44100)
            output_path: Path to the final output file with audio
            progress_callback: Optional callback function to report progress (0.0 to 1.0)
        
        Returns:
            True if successful, False otherwise
        """
        if not FFMPEG_AVAILABLE:
            _missing = []
            if not _FFMPEG_PYTHON_AVAILABLE:
                _missing.append("ffmpeg-python")
            if not _SOUNDFILE_AVAILABLE:
                _missing.append("soundfile")
            logger.warning(
                "%s not installed; audio merge skipped, saving video without audio. "
                "Fix: pip install %s",
                " and ".join(_missing), " ".join(_missing),
            )
            return False
        
        try:
            # Verify video file exists
            if not os.path.exists(video_path):
                logger.error("Video file not found for merge: %s", video_path)
                return False
            
            # Report progress: Starting concatenation
            if progress_callback:
                progress_callback(0.1)
            
            # Validate and filter audio samples
            if not audio_samples:
                logger.warning("No audio samples collected for merge")
                return False
            
            # Filter out empty or invalid arrays
            valid_samples = [sample for sample in audio_samples 
                           if isinstance(sample, np.ndarray) and sample.size > 0]
            
            if not valid_samples:
                logger.warning("No valid audio samples to merge")
                return False
            
            # Concatenate all valid audio samples
            full_audio = np.concatenate(valid_samples)
            
            # ---- Trailing audio trim ----
            # Trim the audio track to match the video duration (derived from the
            # video file's frame count and fps).  This prevents the audio from
            # overrunning the video when extra step_duration chunks were collected
            # after the last video frame, which would cause the player to freeze
            # on the last frame while extra audio plays out.
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    if video_fps > 0 and video_frame_count > 0:
                        video_duration_s = video_frame_count / video_fps
                        max_audio_samples = int(video_duration_s * sample_rate)
                        if len(full_audio) > max_audio_samples > 0:
                            trimmed = len(full_audio) - max_audio_samples
                            full_audio = full_audio[:max_audio_samples]
                            logger.info(
                                "Trimmed %d trailing audio samples (%.1f ms) "
                                "to match video duration %.3f s",
                                trimmed,
                                trimmed / sample_rate * 1000.0,
                                video_duration_s,
                            )
                else:
                    cap.release()
            except Exception as e:
                logger.debug("Could not trim trailing audio: %s", e)
            
            # Report progress: Audio concatenated
            if progress_callback:
                progress_callback(0.3)
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Write audio to temporary WAV file
                sf.write(temp_audio_path, full_audio, sample_rate)
                
                # Report progress: Audio file written
                if progress_callback:
                    progress_callback(0.5)
                
                # Use ffmpeg to merge video and audio
                logger.info(
                    "Merging final video with audio: video=%s samples=%d sample_rate=%s output=%s",
                    video_path,
                    len(full_audio),
                    sample_rate,
                    output_path,
                )
                video_input = ffmpeg.input(video_path)
                audio_input = ffmpeg.input(temp_audio_path)
                
                # Merge video and audio streams.
                # Use -shortest so that if the audio track is marginally longer
                # than the video (e.g. one extra step_duration chunk due to the
                # off-by-one at the last frame boundary), the output is trimmed
                # to the video duration — preventing the image from freezing on
                # the last frame while the extra audio plays out.
                output = ffmpeg.output(
                    video_input,
                    audio_input,
                    output_path,
                    vcodec='copy',  # Copy video codec (no re-encoding)
                    acodec='aac',   # Use AAC for audio (widely compatible)
                    loglevel='error',  # Only show errors
                    shortest=None,  # Trim output to the shorter of the two streams
                )
                
                # Overwrite output file if it exists
                output = ffmpeg.overwrite_output(output)
                
                # Report progress: Starting ffmpeg merge
                if progress_callback:
                    progress_callback(0.7)
                
                # Run ffmpeg – use the resolved executable so the merge
                # works even when 'ffmpeg' is not on the system PATH
                # (e.g. imageio-ffmpeg bundled binary on Windows).
                ffmpeg.run(
                    output,
                    cmd=_get_ffmpeg_exe(),
                    capture_stdout=True,
                    capture_stderr=True,
                )
                
                # Report progress: Merge complete
                if progress_callback:
                    progress_callback(1.0)
                
                logger.info("Successfully merged audio and video to %s", output_path)
                return True
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                    
        except Exception as e:
            logger.exception("Error merging audio and video: %s", e)
            return False

    def close(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag
        
        try:
            # Wait for any ongoing merge threads to complete
            if tag_node_name in self._merge_threads_dict:
                try:
                    thread = self._merge_threads_dict[tag_node_name]
                    if thread.is_alive():
                        logger.info("Waiting for merge completion for %s", tag_node_name)
                        thread.join(timeout=30)  # Wait up to 30 seconds
                except Exception as e:
                    logger.warning("Error waiting for merge thread: %s", e)
                finally:
                    self._merge_threads_dict.pop(tag_node_name, None)
            
            # Clean up merge progress
            if tag_node_name in self._merge_progress_dict:
                self._merge_progress_dict.pop(tag_node_name, None)

            # Release on-disk writer (PyAVEncoder or cv2)
            if tag_node_name in self._ondisk_writers_dict:
                try:
                    od = self._ondisk_writers_dict[tag_node_name]
                    if isinstance(od, cv2.VideoWriter):
                        od.release()
                    else:
                        od.close()
                except Exception as e:
                    logger.warning("Error closing on-disk writer in close: %s", e)
                finally:
                    self._ondisk_writers_dict.pop(tag_node_name, None)

            # Release video writer
            if tag_node_name in self._video_writer_dict:
                try:
                    self._video_writer_dict[tag_node_name].release()
                except Exception as e:
                    logger.warning("Error releasing video writer in close: %s", e)
                finally:
                    self._video_writer_dict.pop(tag_node_name, None)
            
            # Clean up MKV metadata if exists
            if tag_node_name in self._mkv_metadata_dict:
                try:
                    metadata = self._mkv_metadata_dict[tag_node_name]
                    self._close_metadata_handles(metadata)
                except Exception as e:
                    logger.warning("Error closing metadata in close: %s", e)
                finally:
                    self._mkv_metadata_dict.pop(tag_node_name, None)
                    
        except Exception as e:
            logger.exception("Unexpected error in close method: %s", e)

    def get_setting_dict(self, node_id):
        tag_node_name = str(node_id) + ':' + self.node_tag

        pos = dpg.get_item_pos(tag_node_name)

        setting_dict = {}
        setting_dict['ver'] = self._ver
        setting_dict['pos'] = pos
        setting_dict[tag_node_name + ':WriteMode'] = (
            dpg_get_value(tag_node_name + ':WriteMode') or WRITE_MODE_STANDARD
        )

        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        tag_node_name = str(node_id) + ':' + self.node_tag
        write_mode = setting_dict.get(tag_node_name + ':WriteMode', WRITE_MODE_STANDARD)
        try:
            dpg_set_value(tag_node_name + ':WriteMode', write_mode)
        except Exception:
            pass

    def _async_merge_thread(self, tag_node_name, temp_path, audio_samples, sample_rate, final_path):
        """
        Thread worker function to merge audio and video asynchronously.
        This runs in a separate thread to prevent UI freezing.
        """
        def progress_callback(progress):
            """Update progress in the shared dict"""
            self._merge_progress_dict[tag_node_name] = progress
        
        try:
            # Initialize progress
            self._merge_progress_dict[tag_node_name] = 0.0
            
            # Wait for video file to be fully written (with timeout)
            elapsed = 0
            while not os.path.exists(temp_path) and elapsed < self._FILE_WAIT_TIMEOUT:
                time.sleep(self._FILE_WAIT_INTERVAL)
                elapsed += self._FILE_WAIT_INTERVAL
            
            if not os.path.exists(temp_path):
                logger.error("Temporary video file not found before merge: %s", temp_path)
                raise FileNotFoundError(f"Temporary video file not found: {temp_path}")
            
            # Additional small wait to ensure file is fully flushed
            time.sleep(self._FILE_FLUSH_DELAY)
            
            # Perform the merge with progress reporting
            success = self._merge_audio_video_ffmpeg(
                temp_path,
                audio_samples,
                sample_rate,
                final_path,
                progress_callback=progress_callback
            )
            
            if success:
                # Remove temporary video file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                logger.info("Final video with audio saved: %s", final_path)
            else:
                # If merge failed, rename temp file to final name
                if os.path.exists(temp_path):
                    os.rename(temp_path, final_path)
                logger.warning("Audio merge failed; video without audio saved: %s", final_path)
                
        except Exception as e:
            logger.exception("Error in async merge thread: %s", e)
            # Try to save the temp file as final on error
            if os.path.exists(temp_path):
                try:
                    os.rename(temp_path, final_path)
                    logger.warning("Video saved without merge fallback: %s", final_path)
                except Exception as rename_error:
                    logger.error("Error renaming temp file after merge failure: %s", rename_error)
        finally:
            # Clean up merge progress indicator
            if tag_node_name in self._merge_progress_dict:
                # Set to 1.0 to indicate completion before cleanup
                self._merge_progress_dict[tag_node_name] = 1.0
    


    def _recording_button(self, sender, data, user_data):
        tag_node_name = user_data
        tag_node_button_value_name = tag_node_name + ':' + self.TYPE_TEXT + ':ButtonValue'

        label = dpg.get_item_label(tag_node_button_value_name)

        if label == self._start_label:

            datetime_now = datetime.datetime.now()
            
            startup_time_text = datetime_now.strftime('%Y%m%d_%H%M%S')
            writer_width = self._opencv_setting_dict['video_writer_width']
            writer_height = self._opencv_setting_dict['video_writer_height']
            writer_fps = self._opencv_setting_dict['video_writer_fps']
            video_writer_directory = self._opencv_setting_dict[
                'video_writer_directory']

            # In frozen (onefile) mode, resolve relative paths against the
            # persistent app directory (next to the .exe), not the temp
            # extraction folder.
            if not os.path.isabs(video_writer_directory):
                from src.utils.paths import get_app_dir
                video_writer_directory = os.path.join(
                    get_app_dir(), video_writer_directory.lstrip('./\\'))

            os.makedirs(video_writer_directory, exist_ok=True)

            # Get selected format and write mode
            format_tag = tag_node_name + ':Format'
            video_format = dpg_get_value(format_tag)
            write_mode = dpg_get_value(tag_node_name + ':WriteMode') or WRITE_MODE_STANDARD

            already_recording = (
                tag_node_name in self._video_writer_dict
                or tag_node_name in self._ondisk_writers_dict
            )
            if not already_recording:
                # Determine file extension based on format
                _ext_map = {'AVI': '.avi', 'MKV': '.mkv', 'MP4': '.mp4'}
                ext = _ext_map.get(video_format, '.mp4')
                file_path = os.path.join(video_writer_directory,
                                         f'{startup_time_text}{ext}')

                if write_mode == WRITE_MODE_ON_DISK:
                    # -------------------------------------------------------
                    # On-disk mode: stream frames straight to the final file
                    # via PyAVEncoder (libx264) when available, or via a cv2
                    # writer otherwise.
                    # No audio is collected; RAM usage stays constant.
                    # -------------------------------------------------------
                    if _AV_AVAILABLE:
                        try:
                            od_writer = PyAVEncoder(
                                output_path=file_path,
                                fps=float(writer_fps),
                                frame_size=(writer_width, writer_height),
                                codec="libx264",
                                include_audio=False,
                            )
                            self._ondisk_writers_dict[tag_node_name] = od_writer
                            logger.info(
                                "VideoWriter[%s] ON-DISK recording started "
                                "(PyAVEncoder/libx264) → %s",
                                tag_node_name, file_path,
                            )
                        except Exception as e:
                            logger.warning(
                                "PyAVEncoder init failed (%s); falling back to cv2.", e
                            )
                            od_writer = None
                    else:
                        od_writer = None

                    if od_writer is None:
                        # cv2 fallback
                        if video_format == 'AVI':
                            cv2_codec = 'XVID'
                        elif video_format == 'MKV':
                            cv2_codec = 'FFV1'
                        else:
                            cv2_codec = 'mp4v'
                        cv2_od = cv2.VideoWriter(
                            file_path,
                            cv2.VideoWriter_fourcc(*cv2_codec),
                            writer_fps,
                            (writer_width, writer_height),
                        )
                        self._ondisk_writers_dict[tag_node_name] = cv2_od
                        logger.info(
                            "VideoWriter[%s] ON-DISK recording started "
                            "(cv2/%s) → %s",
                            tag_node_name, cv2_codec, file_path,
                        )

                    self._frame_counter_dict[tag_node_name] = 0
                    self._recording_metadata_dict[tag_node_name] = {
                        'final_path': file_path,
                        'temp_path': None,
                        'format': video_format,
                        'fps': float(writer_fps),
                        'mode': WRITE_MODE_ON_DISK,
                    }

                else:
                    # -------------------------------------------------------
                    # Standard mode: cv2.VideoWriter → temp file; audio
                    # buffered in RAM; async merge at the end.
                    # -------------------------------------------------------
                    if video_format == 'AVI':
                        cv2_codec = 'MJPG'
                    elif video_format == 'MKV':
                        cv2_codec = 'FFV1'
                    else:
                        cv2_codec = 'mp4v'

                    temp_file_path = os.path.join(
                        video_writer_directory, f'{startup_time_text}_temp{ext}'
                    )
                    self._video_writer_dict[tag_node_name] = cv2.VideoWriter(
                        temp_file_path,
                        cv2.VideoWriter_fourcc(*cv2_codec),
                        writer_fps,
                        (writer_width, writer_height),
                    )

                    # Initialize metadata tracking for MKV
                    if video_format == 'MKV':
                        self._mkv_metadata_dict[tag_node_name] = {
                            'audio_handles': {},
                            'json_handles': {},
                            'file_path': file_path,
                        }
                        metadata_dir = os.path.join(
                            video_writer_directory, f'{startup_time_text}_metadata'
                        )
                        os.makedirs(metadata_dir, exist_ok=True)

                    # Initialize audio sample collection
                    self._audio_samples_dict[tag_node_name] = []
                    self._last_chunk_index_dict[tag_node_name] = -1

                    # Initialise per-node frame counter and SyncVideoWriter
                    self._frame_counter_dict[tag_node_name] = 0
                    self._sync_writers_dict[tag_node_name] = SyncVideoWriter(
                        fps=float(writer_fps),
                        max_buffer_size=max(4, int(writer_fps * 0.2)),
                    )

                    self._recording_metadata_dict[tag_node_name] = {
                        'final_path': file_path,
                        'temp_path': temp_file_path,
                        'format': video_format,
                        'sample_rate': 22050,
                        'fps': float(writer_fps),
                        'mode': WRITE_MODE_STANDARD,
                    }
                    logger.info(
                        "VideoWriter[%s] STANDARD recording started "
                        "format=%s codec=%s fps=%s size=%sx%s "
                        "temp=%s final=%s",
                        tag_node_name, video_format, cv2_codec,
                        writer_fps, writer_width, writer_height,
                        temp_file_path, file_path,
                    )

            dpg.set_item_label(tag_node_button_value_name, self._stop_label)
        elif label == self._stop_label:
            try:
                # --- On-disk mode stop ---
                if tag_node_name in self._ondisk_writers_dict:
                    od_writer = self._ondisk_writers_dict.pop(tag_node_name)
                    try:
                        if isinstance(od_writer, cv2.VideoWriter):
                            od_writer.release()
                        else:
                            od_writer.close()
                    except Exception as e:
                        logger.exception("Error closing on-disk writer: %s", e)
                    self._frame_counter_dict.pop(tag_node_name, None)
                    meta = self._recording_metadata_dict.pop(tag_node_name, {})
                    logger.info(
                        "VideoWriter[%s] ON-DISK recording stopped → %s",
                        tag_node_name, meta.get('final_path', '?'),
                    )
                    try:
                        if dpg.does_item_exist(tag_node_button_value_name):
                            dpg.set_item_label(tag_node_button_value_name, self._start_label)
                    except Exception:
                        pass
                    return  # nothing more to do for on-disk mode

                # --- Standard mode stop ---
                # Flush remaining frames from the sync writer before releasing cv2 writer
                if tag_node_name in self._sync_writers_dict:
                    sync_writer = self._sync_writers_dict.pop(tag_node_name)
                    cv2_writer = self._video_writer_dict.get(tag_node_name)
                    if cv2_writer is not None:
                        sync_writer.flush_and_collect()  # drain heap; frames already written via flush_ready
                self._frame_counter_dict.pop(tag_node_name, None)

                # Release video writer and ensure file is flushed to disk
                if tag_node_name in self._video_writer_dict:
                    try:
                        self._video_writer_dict[tag_node_name].release()
                    except Exception as e:
                        logger.exception("Error releasing video writer: %s", e)
                    finally:
                        # Always remove from dict even if release fails
                        self._video_writer_dict.pop(tag_node_name, None)
                
                # Merge audio and video if audio samples were collected
                if tag_node_name in self._audio_samples_dict and len(self._audio_samples_dict[tag_node_name]) > 0:
                    if tag_node_name in self._recording_metadata_dict:
                        try:
                            metadata = self._recording_metadata_dict[tag_node_name]
                            temp_path = metadata['temp_path']
                            final_path = metadata['final_path']
                            sample_rate = metadata['sample_rate']
                            
                            # Copy audio samples for the thread (to avoid race conditions)
                            audio_samples_copy = copy.deepcopy(self._audio_samples_dict[tag_node_name])
                            total_samples = sum(
                                len(s) for s in audio_samples_copy if hasattr(s, '__len__')
                            )
                            duration_s = total_samples / max(sample_rate, 1)
                            
                            # Start merge in a separate thread to prevent UI freezing
                            logger.info(
                                "VideoWriter[%s]: Stopping recording. "
                                "Audio collected: %d chunks, ~%d samples, SR=%s Hz (~%.1fs). "
                                "Starting async merge → %s",
                                tag_node_name,
                                len(audio_samples_copy),
                                total_samples,
                                sample_rate,
                                duration_s,
                                final_path,
                            )
                            merge_thread = threading.Thread(
                                target=self._async_merge_thread,
                                args=(tag_node_name, temp_path, audio_samples_copy, sample_rate, final_path),
                                daemon=True
                            )
                            merge_thread.start()
                            
                            # Store thread reference for tracking
                            self._merge_threads_dict[tag_node_name] = merge_thread
                        except Exception as e:
                            logger.exception("Error starting audio/video merge: %s", e)
                        finally:
                            # Clean up metadata
                            self._recording_metadata_dict.pop(tag_node_name, None)
                else:
                    # No audio samples, just rename temp file to final name
                    if tag_node_name in self._recording_metadata_dict:
                        try:
                            metadata = self._recording_metadata_dict[tag_node_name]
                            temp_path = metadata['temp_path']
                            final_path = metadata['final_path']
                            n_frames = self._frame_counter_dict.get(tag_node_name, 0)
                            logger.warning(
                                "VideoWriter[%s]: Stopping recording with NO audio chunks "
                                "(recorded %d frames). Video will be saved without audio. "
                                "Check that the Video node 'Frames only' checkbox is unchecked.",
                                tag_node_name, n_frames,
                            )
                            
                            if os.path.exists(temp_path):
                                os.rename(temp_path, final_path)
                                logger.info("Video saved without audio: %s", final_path)
                            else:
                                logger.warning("Temporary video file not found while stopping recording: %s", temp_path)
                        except Exception as e:
                            logger.exception("Error saving video file: %s", e)
                        finally:
                            self._recording_metadata_dict.pop(tag_node_name, None)
                
                # Clean up audio samples
                if tag_node_name in self._audio_samples_dict:
                    self._audio_samples_dict.pop(tag_node_name, None)
                self._last_chunk_index_dict.pop(tag_node_name, None)
                
                # Close metadata file handles if MKV
                if tag_node_name in self._mkv_metadata_dict:
                    try:
                        metadata = self._mkv_metadata_dict[tag_node_name]
                        self._close_metadata_handles(metadata)
                    except Exception as e:
                        logger.exception("Error closing metadata handles: %s", e)
                    finally:
                        self._mkv_metadata_dict.pop(tag_node_name, None)

                # Always update button label, even if errors occurred
                try:
                    if dpg.does_item_exist(tag_node_button_value_name):
                        dpg.set_item_label(tag_node_button_value_name, self._start_label)
                except Exception as e:
                    logger.warning("Error updating button label: %s", e)
                    
            except Exception as e:
                # Catch-all for any unexpected errors
                logger.exception("Unexpected error while stopping video recording: %s", e)
                # Try to update button label anyway
                try:
                    if dpg.does_item_exist(tag_node_button_value_name):
                        dpg.set_item_label(tag_node_button_value_name, self._start_label)
                except Exception:
                    # Ignore errors in final cleanup attempt
                    pass
