#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLOX distillation node.

Detects objects with a small YOLOX student while a large YOLOX teacher trains
it in the background on the frames going through the node.

Only the student runs in the pipeline.  Collection of teacher predictions,
training, validation and deployment of better students are done by
``DistillationEngine`` in its own threads; see docs/DISTILLATION_YOLOX.md.
"""
import math
import os
import sys
import time

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from node_editor.util import dpg_get_value, dpg_set_value, run_unlocked
from node.basenode import Node
from node.DLNode.object_detection.coco_class_names import coco_class_names
from node.DLNode.online_training.distill.engine import DistillationEngine
from node.DLNode.online_training.distill.students import (
    DEFAULT_STUDENT,
    describe_student,
    find_yolox_models,
    resolve_student,
    student_choices,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

_STATUS_PERIOD = 0.5  # seconds between two refreshes of the status text
_SETTINGS = (':Teacher', ':Student', ':Train', ':ChartMetrics', ':Score')  # saved with the graph

__all__ = ['FactoryNode', 'Node', 'distillation_metrics', 'find_yolox_models', 'format_status']


def _work_root():
    if getattr(sys, 'frozen', False):
        from src.utils.paths import get_app_dir
        return os.path.join(get_app_dir(), 'distillation')
    return os.path.abspath('_Distillation')


def _students_dir():
    """Where the resized students of the catalogue are generated."""
    return os.path.join(_work_root(), 'students')


def format_status(status):
    """Three short lines describing what the engine is doing."""
    phase = status['phase']
    if not status['enabled'] and phase not in ('error', 'unavailable'):
        phase = 'paused'
    device = f" ({status['device']})" if status.get('device') else ''
    lines = [
        f"student v{status['version']} - {phase}{device}",
        f"samples: {status['train_samples']} train / {status['val_samples']} val",
    ]
    if phase in ('error', 'unavailable'):
        lines.append(status['message'][:60])
    elif status.get('deployed'):
        deployed = status['deployed']
        lines.append(f"val loss {deployed['loss']:.3f}  AP50 {deployed['ap50']:.2f}")
    elif status.get('train_loss') is not None:
        lines.append(f"train loss {status['train_loss']:.3f}")
    else:
        lines.append('waiting for samples')
    return '\n'.join(lines)


def _number(value):
    """``value`` if it is a finite number, else ``None`` (not measured)."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def distillation_metrics(status, detections, inference_ms):
    """Flat numeric metrics of the distillation, as plotted by the Chart node.

    ``val_*`` describe the deployed student on the validation window (the
    most recent samples, never trained on), against the teacher.
    ``original_val_*`` are the same measures for the student the distillation
    started from, on the same images, and ``f1_gain`` is their difference:
    what the distillation has brought so far.  ``candidate_val_*`` describe
    the student trained by the last round, deployed or not.
    ``disagreement`` (0 = identical) compares teacher and deployed student on
    the last frame shown to the teacher.

    A value is ``None`` until it has been measured.  The Chart node shows the
    first key by default.
    """
    deployed = status.get('deployed') or {}
    original = status.get('original') or {}
    candidate = status.get('candidate') or {}
    val_f1, original_f1 = _number(deployed.get('f1')), _number(original.get('f1'))
    return {
        'val_f1': val_f1,
        'val_ap50': _number(deployed.get('ap50')),
        'val_precision': _number(deployed.get('precision')),
        'val_recall': _number(deployed.get('recall')),
        'val_loss': _number(deployed.get('loss')),
        'original_val_f1': original_f1,
        'original_val_ap50': _number(original.get('ap50')),
        'f1_gain': None if val_f1 is None or original_f1 is None else val_f1 - original_f1,
        'candidate_val_loss': _number(candidate.get('loss')),
        'candidate_val_ap50': _number(candidate.get('ap50')),
        'train_loss': _number(status.get('train_loss')),
        'disagreement': _number(status.get('disagreement')),
        'version': status['version'],
        'rounds': status['rounds'],
        'train_samples': status['train_samples'],
        'val_samples': status['val_samples'],
        'detections': int(detections),
        'inference_ms': _number(inference_ms),
    }


class FactoryNode:
    node_label = 'YoloxDistillation'
    node_tag = 'YoloxDistillation'

    def add_node(
        self,
        parent,
        node_id,
        pos=[0, 0],
        opencv_setting_dict=None,
        callback=None,
    ):
        node = Node()
        name = node.tag_node_name = str(node_id) + ':' + self.node_tag
        node._opencv_setting_dict = opencv_setting_dict

        tag_input = name + ':' + node.TYPE_IMAGE + ':Input01'
        tag_output_image = name + ':' + node.TYPE_IMAGE + ':Output01'
        tag_output_json = name + ':' + node.TYPE_JSON + ':Output02'

        width = opencv_setting_dict['process_width']
        height = opencv_setting_dict['process_height']
        texture = node.convert_cv_to_dpg(np.zeros((height, width, 3)), width, height)
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width, height, texture,
                tag=tag_output_image + 'Value',
                format=dpg.mvFormat_Float_rgb,
            )

        models = find_yolox_models()
        teachers = list(models)
        students = student_choices(models)
        default_student = DEFAULT_STUDENT if DEFAULT_STUDENT in students else (
            students[0] if students else ''
        )
        with dpg.node(tag=name, parent=parent, label=node.node_label, pos=pos):
            with dpg.node_attribute(tag=tag_input, attribute_type=dpg.mvNode_Attr_Input):
                dpg.add_text(tag=tag_input + 'Value', default_value='Image')

            with dpg.node_attribute(tag=tag_output_image, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_image(tag_output_image + 'Value')

            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_combo(
                    teachers, tag=name + ':Teacher', label='teacher', width=width - 80,
                    default_value=teachers[-1] if teachers else '',
                )
                dpg.add_combo(
                    students, tag=name + ':Student', label='student', width=width - 80,
                    default_value=default_student,
                    callback=lambda sender, value: dpg_set_value(
                        name + ':StudentInfo', describe_student(value)
                    ),
                )
                dpg.add_text(
                    tag=name + ':StudentInfo', default_value=describe_student(default_student),
                    wrap=width - 10,
                )
                dpg.add_checkbox(tag=name + ':Train', label='Distill (background)',
                                 default_value=False)
                # Unchecked, a Chart node connected to the JSON output counts
                # the detected classes instead of plotting the metrics
                dpg.add_checkbox(tag=name + ':ChartMetrics', label='Metrics to Chart',
                                 default_value=True)
                dpg.add_slider_float(
                    tag=name + ':Score', label='score', width=width - 80,
                    default_value=0.3, min_value=0.05, max_value=0.95,
                )
                dpg.add_text(tag=name + ':Status', default_value='no image yet')
                with dpg.group(horizontal=True):
                    dpg.add_button(label='Export ONNX', callback=lambda: node.export_student())
                    dpg.add_button(label='Reset student', callback=lambda: node.reset_student())

            with dpg.node_attribute(tag=tag_output_json, attribute_type=dpg.mvNode_Attr_Output):
                dpg.add_text(tag=tag_output_json + 'Value', default_value='JSON')

        return node


class Node(Node):
    _ver = '0.0.1'

    node_label = 'YoloxDistillation'
    node_tag = 'YoloxDistillation'

    _opencv_setting_dict = None

    # Engines are shared by the nodes using the same teacher/student pair:
    # two engines must not write in the same directory.
    _engines = {}

    def __init__(self):
        self._engine = None
        self._engine_key = None
        self._failed_key = None
        self._message = ''
        self._last_status = 0.0

    # ------------------------------------------------------------------ #

    def _release_engine(self):
        if self._engine is None:
            return
        entry = Node._engines.get(self._engine_key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] <= 0:
                entry[0].stop()
                del Node._engines[self._engine_key]
        self._engine = self._engine_key = None

    def _get_engine(self, teacher, student):
        key = (teacher, student)
        if key == self._engine_key:
            return self._engine
        if key == self._failed_key:
            return None
        self._release_engine()
        models = find_yolox_models()
        try:
            if teacher not in models or not student:
                raise ValueError('select a teacher and a student model')
            entry = Node._engines.get(key)
            if entry is None:
                # Catalogue students at a reduced input size are generated
                # from the shipped file the first time they are selected
                student_path = resolve_student(student, _students_dir(), models)
                workdir = os.path.join(_work_root(), f'{student}__from__{teacher}')
                engine = DistillationEngine(models[teacher], student_path, workdir)
                engine.start()
                entry = Node._engines[key] = [engine, 0]
            entry[1] += 1
            self._engine, self._engine_key = entry[0], key
            self._failed_key, self._message = None, ''
        except Exception as exc:
            logger.error('[YoloxDistillation] cannot use %s -> %s: %s', teacher, student, exc)
            self._failed_key, self._message = key, str(exc)
        return self._engine

    def export_student(self):
        if self._engine is None:
            return
        try:
            directory = os.path.join(_work_root(), 'exports')
            os.makedirs(directory, exist_ok=True)
            teacher, student = self._engine_key
            path = os.path.join(
                directory, f'{student}_distilled_v{self._engine.version}.onnx'
            )
            self._engine.export(path)
            self._message = f'exported: {path}'
            logger.info('[YoloxDistillation] student exported to %s', path)
        except Exception as exc:
            self._message = f'export failed: {exc}'
            logger.error('[YoloxDistillation] export failed', exc_info=True)

    def reset_student(self):
        if self._engine is not None:
            self._engine.reset()
            self._message = 'student reset to the original model'

    # ------------------------------------------------------------------ #

    def update(self, node_id, connection_list, node_image_dict, node_result_dict, node_audio_dict):
        name = str(node_id) + ':' + self.node_tag
        tag_output_image = name + ':' + self.TYPE_IMAGE + ':Output01Value'

        frame = self.get_input_frame(connection_list, node_image_dict, node_audio_dict)
        if frame is None:
            return {'image': None, 'json': None, 'audio': None}

        engine = self._get_engine(dpg_get_value(name + ':Teacher'), dpg_get_value(name + ':Student'))
        if engine is None:
            dpg_set_value(name + ':Status', (self._message or 'no model')[:120])
            return {'image': frame, 'json': None, 'audio': None}

        score_th = float(dpg_get_value(name + ':Score') or 0.3)
        engine.config.score_th = score_th
        engine.set_training(bool(dpg_get_value(name + ':Train')))

        # The student is the only model run in the pipeline; the frame is
        # merely offered to the engine, which keeps one every now and then.
        started = time.perf_counter()
        boxes, scores, class_ids = run_unlocked(engine.detect, frame, score_th)
        inference_ms = (time.perf_counter() - started) * 1000.0
        engine.submit(frame)

        class_names = (
            coco_class_names if engine.num_classes == len(coco_class_names)
            else {i: f'class_{i}' for i in range(engine.num_classes)}
        )
        output = frame.copy()
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output, f'{class_names.get(int(class_id), class_id)} {score:.2f}',
                (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        width = self._opencv_setting_dict['process_width']
        height = self._opencv_setting_dict['process_height']
        dpg_set_value(tag_output_image, self.convert_cv_to_dpg(output, width, height))

        status = engine.status()
        now = time.monotonic()
        if now - self._last_status >= _STATUS_PERIOD:
            self._last_status = now
            text = format_status(status)
            if self._message:
                text += '\n' + self._message[-60:]
            dpg_set_value(name + ':Status', text)

        metrics = distillation_metrics(status, len(boxes), inference_ms)
        result = {
            'bboxes': boxes.tolist(),
            'scores': scores.tolist(),
            'class_ids': class_ids.tolist(),
            'class_names': class_names,
            'score_th': score_th,
            'distillation': {'phase': status['phase'], **metrics},
        }
        # The key the Chart node plots as time series (as for OnlineTraining)
        if dpg_get_value(name + ':ChartMetrics') is not False:
            result['distillation_losses'] = metrics
        return {'image': output, 'json': result, 'audio': None}

    def close(self, node_id):
        self._release_engine()

    def get_setting_dict(self, node_id):
        name = str(node_id) + ':' + self.node_tag
        setting_dict = {'ver': self._ver, 'pos': dpg.get_item_pos(name)}
        for suffix in _SETTINGS:
            setting_dict[name + suffix] = dpg_get_value(name + suffix)
        return setting_dict

    def set_setting_dict(self, node_id, setting_dict):
        name = str(node_id) + ':' + self.node_tag
        for suffix in _SETTINGS:
            if name + suffix in setting_dict:
                dpg_set_value(name + suffix, setting_dict[name + suffix])
        if name + ':Student' in setting_dict:
            dpg_set_value(name + ':StudentInfo', describe_student(setting_dict[name + ':Student']))
