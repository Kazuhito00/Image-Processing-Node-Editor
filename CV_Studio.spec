# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for CV_Studio

This spec file builds a standalone .exe for CV_Studio with:
- All nodes (Input, Process, DL, Audio, etc.)
- ONNX models for object detection
- DearPyGUI resources
- Fonts and configuration files
- All required Python dependencies

Usage:
    pyinstaller CV_Studio.spec

The .exe will be created in the 'dist/CV_Studio' directory.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the base directory
block_cipher = None
base_path = os.path.abspath('.')

# Collect all submodules for key packages
hiddenimports = []
hiddenimports += collect_submodules('dearpygui')
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('onnxruntime')
hiddenimports += collect_submodules('mediapipe')
hiddenimports += collect_submodules('numpy')
hiddenimports += collect_submodules('librosa')
hiddenimports += collect_submodules('soundfile')
hiddenimports += collect_submodules('sounddevice')
hiddenimports += collect_submodules('matplotlib')
hiddenimports += collect_submodules('scipy')
hiddenimports += collect_submodules('sklearn')
hiddenimports += collect_submodules('pafy')
hiddenimports += collect_submodules('youtube_dl')
hiddenimports += collect_submodules('yt_dlp')
hiddenimports += collect_submodules('filterpy')
hiddenimports += collect_submodules('pymongo')
hiddenimports += collect_submodules('bson')
hiddenimports += collect_submodules('pytz')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('requests')
hiddenimports += collect_submodules('serial')
hiddenimports += collect_submodules('rich')
hiddenimports += collect_submodules('lap')
hiddenimports += collect_submodules('motpy')
hiddenimports += collect_submodules('norfair')
hiddenimports += collect_submodules('ffmpeg')
hiddenimports += collect_submodules('streamlink')
hiddenimports += collect_submodules('wordcloud')
hiddenimports += collect_submodules('wsdiscovery')
hiddenimports += collect_submodules('onvif')
hiddenimports += collect_submodules('zeep')
hiddenimports += collect_submodules('lxml')
hiddenimports += collect_submodules('isodate')
hiddenimports += collect_submodules('ifaddr')
# YOLOX distillation: the student ONNX is converted to PyTorch for training
hiddenimports += collect_submodules('onnx2torch')
hiddenimports += ['onnx', 'onnx.numpy_helper', 'onnx.shape_inference']

# Add explicit hidden imports for node modules
hiddenimports += [
    'node',
    'node.InputNode',
    'node.ProcessNode',
    'node.DLNode',
    'node.AudioProcessNode',
    'node.AudioModelNode',
    'node.StatsNode',
    'node.TimeseriesNode',
    'node.TriggerNode',
    'node.RouterNode',
    'node.ActionNode',
    'node.SystemNode',
    'node.OverlayNode',
    'node.TrackerNode',
    'node.VisualNode',
    'node.VideoNode',
    'node.timestamped_queue',
    'node.queue_adapter',
    'node.basenode',
    'node_editor',
    'node_editor.node_main',
    'node_editor.util',
    'node_editor.style',
    'src',
    'src.utils',
    'src.utils.logging',
    'src.utils.gpu_utils',
    'src.core',
    # Third-party packages
    'pafy',
    'youtube_dl',
    'yt_dlp',
    'filterpy',
    'filterpy.kalman',
    'filterpy.common',
    'pymongo',
    'bson',
    'bson.objectid',
    'pytz',
    'dnspython',
    'PIL',
    'PIL.Image',
    'PIL.ImageGrab',
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    'requests',
    'requests.adapters',
    'requests.auth',
    'scipy',
    'scipy.spatial',
    'scipy.linalg',
    'sklearn',
    'sklearn.metrics',
    'sklearn.preprocessing',
    'rich',
    'rich.console',
    'rich.progress',
    'lap',
    'motpy',
    'norfair',
    'ffmpeg',
    'sounddevice',
    'streamlink',
    'wordcloud',
    # ONVIF / WS-Discovery (lazy imports in node_scan.py)
    'wsdiscovery',
    'wsdiscovery.discovery',
    'wsdiscovery.scope',
    'wsdiscovery.qname',
    'onvif',
    'onvif.client',
    'onvif.exceptions',
    'onvif.definition',
    'zeep',
    'zeep.transports',
    'zeep.wsdl',
    'lxml',
    'lxml.etree',
    'isodate',
    'ifaddr',
]

# Collect data files
datas = []

# Add node directory with all subdirectories and files
datas.append(('node', 'node'))

# Add node_editor directory
datas.append(('node_editor', 'node_editor'))

# Add src directory
datas.append(('src', 'src'))

# ONNX models are automatically included via the 'node' directory above
# The entire node directory is copied recursively, including:
# - All .onnx model files in node/DLNode/*/model/
# - All node Python modules and supporting files
# This ensures all ONNX models for object detection are bundled

# Note: Font and setting files are already included in the node_editor directory above
# No need to add them separately as it can cause PyInstaller bundling conflicts

# Collect data files from packages that need them
datas += collect_data_files('dearpygui')
datas += collect_data_files('mediapipe')
datas += collect_data_files('onnxruntime')
datas += collect_data_files('librosa')
datas += collect_data_files('sklearn')
datas += collect_data_files('streamlink')
datas += collect_data_files('wordcloud')
datas += collect_data_files('zeep')

# ONVIF WSDL files - onvif-zeep installs a top-level 'wsdl' folder in
# site-packages that ONVIFCamera needs at runtime.
import importlib.util as _iu
_onvif_spec = _iu.find_spec('onvif')
if _onvif_spec and _onvif_spec.submodule_search_locations:
    import pathlib as _pl
    _onvif_pkg = _pl.Path(list(_onvif_spec.submodule_search_locations)[0])
    _wsdl_dir = _onvif_pkg.parent / 'wsdl'
    if _wsdl_dir.is_dir():
        datas.append((str(_wsdl_dir), 'wsdl'))

# Binary excludes - exclude unnecessary binaries
binaries = []

a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[base_path],
    hooksconfig={},
    runtime_hooks=[os.path.join(base_path, 'hook-runtime-cv-studio.py')],
    excludes=[
        'tkinter',
        'PyQt5',
        'PySide2',
        'PySide6',
        'wx',
        'IPython',
        'jupyter',
        'notebook',
        'sphinx',
        'pytest',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CV_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False to hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon='icon.ico' if you have an icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CV_Studio',
)
