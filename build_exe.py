#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build script for CV_Studio executable

This script automates the process of building a standalone .exe for CV_Studio
using PyInstaller. It includes all necessary dependencies, ONNX models, and
node resources.

Usage:
    python build_exe.py [options]

Options:
    --clean              Clean build directories before building
    --onefile            Create a single executable file (slower startup)
    --windowed           Hide console window (GUI only mode)
    --debug              Build with debug information
    --icon ICON          Path to icon file (.ico)
    --skip-package-check Skip package availability check (useful in CI/CD)
    --help               Show this help message

Examples:
    # Standard build (creates CV_Studio folder with exe and dependencies)
    python build_exe.py

    # Clean build
    python build_exe.py --clean

    # Single file exe (no separate folder, but slower startup)
    python build_exe.py --onefile

    # GUI mode without console window
    python build_exe.py --windowed

    # With custom icon
    python build_exe.py --icon CV_Studio.ico

    # CI/CD build (skip package check, assumes packages are already installed)
    python build_exe.py --clean --skip-package-check
"""

import os
import sys
import shutil
import subprocess
import argparse

# Constants
VCPP_REDISTRIBUTABLE_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"

# Ensure UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import io
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def print_banner():
    """Print build script banner"""
    print("=" * 70)
    print("  CV_Studio - Executable Build Script")
    print("  Building standalone .exe with PyInstaller")
    print("=" * 70)
    print()


def check_requirements(skip_package_check=False):
    """Check if required tools are installed
    
    Args:
        skip_package_check: If True, skip checking for required packages (useful in CI/CD)
        
    Returns:
        bool: True if all requirements are satisfied or skipped, False if checks fail
    """
    print("[1/6] Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7 or higher is required")
        return False
    
    print(f"  ✓ Python {sys.version.split()[0]}")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"  ✓ PyInstaller installed")
    except ImportError:
        print("  ✗ PyInstaller not found")
        print("\nInstalling PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        print("  ✓ PyInstaller installed")
    
    # Skip package check if requested (e.g., in CI/CD where packages are pre-installed)
    if skip_package_check:
        print("  ℹ Skipping package check (--skip-package-check enabled)")
        print()
        return True
    
    # Check required packages
    # Map of package names to their import names
    required_packages = {
        'dearpygui': 'dearpygui',
        'opencv-contrib-python': 'cv2',
        'onnxruntime': 'onnxruntime',
        'numpy': 'numpy',
        'mediapipe': 'mediapipe',
        'scipy': 'scipy',
        'lap': 'lap',
        'motpy': 'motpy',
        'norfair': 'norfair',
        'filterpy': 'filterpy',
        'ffmpeg-python': 'ffmpeg',
        'rich': 'rich',
        'scikit-learn': 'sklearn',
        'pyserial': 'serial',
        'pymongo': 'pymongo',
        'Pillow': 'PIL',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'sounddevice': 'sounddevice',
        'matplotlib': 'matplotlib',
        'requests': 'requests',
        'pafy': 'pafy',
        'yt-dlp': 'yt_dlp',
        'pytz': 'pytz',
        'streamlink': 'streamlink',
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"  ✗ {package_name}")
        except (OSError, RuntimeError) as e:
            # Catch runtime exceptions when importing
            # OSError: Common for DLL loading errors (missing C++ runtime dependencies)
            # RuntimeError: Can occur when package dependencies are incompatible
            missing_packages.append(package_name)
            print(f"  ✗ {package_name} (failed to import: {type(e).__name__})")
    
    if missing_packages:
        print(f"\nWARNING: Missing {len(missing_packages)} package(s):")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print()
        
        # Special handling for onnxruntime errors
        if 'onnxruntime' in missing_packages:
            print("INFO: ONNXRUNTIME TROUBLESHOOTING:")
            print("  onnxruntime requires Visual C++ Redistributable on Windows.")
            print("  If you see errors like 'DLL load failed' or import errors:")
            print()
            print("  Solution 1: Install Visual C++ Redistributable")
            print(f"    Download: {VCPP_REDISTRIBUTABLE_URL}")
            print("    Run the installer and restart your terminal")
            print()
            print("  Solution 2: Use --skip-package-check flag")
            print("    python build_exe.py --skip-package-check")
            print("    (Only if you're sure dependencies are installed)")
            print()
        
        # Check if running in non-interactive environment (CI/CD)
        if not sys.stdin.isatty():
            print("Running in non-interactive mode (CI/CD detected)")
            print("ERROR: Cannot continue with missing packages in non-interactive mode")
            print("\nTo fix this issue:")
            print("  1. Install dependencies: pip install -r requirements.txt")
            print("  2. Or use: python build_exe.py --skip-package-check")
            return False
        
        # Interactive mode: offer to install packages
        print("Options:")
        print("  1. Install missing packages now (recommended)")
        print("  2. Continue without installing (not recommended, build will likely fail)")
        print("  3. Exit and install manually")
        print()
        
        try:
            response = input("Choose option (1/2/3) [1]: ").strip()
            if not response:
                response = '1'
            
            if response == '1':
                # Install missing packages
                print("\nInstalling missing packages from requirements.txt...")
                print("This may take several minutes...\n")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                        check=True
                    )
                    print("\n✓ Packages installed successfully!")
                    print("  Re-checking requirements...\n")
                    
                    # Re-check if packages are now available
                    still_missing = []
                    for package_name, import_name in required_packages.items():
                        try:
                            __import__(import_name)
                        except ImportError:
                            still_missing.append(package_name)
                        except (OSError, RuntimeError) as e:
                            # Handle common runtime exceptions (e.g., DLL loading errors)
                            still_missing.append(package_name)
                            if package_name == 'onnxruntime':
                                print(f"INFO: {package_name} installed but has runtime error: {type(e).__name__}")
                                print(f"  This is usually due to missing Visual C++ Redistributable")
                                print(f"  Download from: {VCPP_REDISTRIBUTABLE_URL}")
                    
                    if still_missing:
                        print(f"WARNING: Missing {len(still_missing)} package(s):")
                        for pkg in still_missing:
                            print(f"  - {pkg}")
                        print("You may need to install them manually or check for installation errors.")
                        if 'onnxruntime' in still_missing:
                            print("\nFor onnxruntime issues:")
                            print("  1. Install Visual C++ Redistributable (see link above)")
                            print("  2. Restart your terminal/command prompt")
                            print("  3. Try running the build again")
                        response = input("\nContinue anyway? (y/N): ")
                        if response.lower() != 'y':
                            return False
                    else:
                        print("✓ All packages are now available!")
                        
                except subprocess.CalledProcessError as e:
                    print(f"\n✗ Failed to install packages (error code {e.returncode})")
                    print("Please install manually with: pip install -r requirements.txt")
                    return False
            elif response == '2':
                print("\nWARNING: Continuing without installing packages.")
                print("The build will likely fail if required modules are not available.")
            elif response == '3':
                print("\nExiting. Install packages with: pip install -r requirements.txt")
                return False
            else:
                print("\nInvalid option. Exiting.")
                return False
                
        except EOFError:
            # Handle EOF error gracefully
            print("\nERROR: Cannot read input (non-interactive environment)")
            print("\nTo fix this issue:")
            print("  1. Install dependencies: pip install -r requirements.txt")
            if 'onnxruntime' in missing_packages:
                print("  2. For onnxruntime: Install Visual C++ Redistributable")
                print(f"     {VCPP_REDISTRIBUTABLE_URL}")
                print("  3. Or use: python build_exe.py --skip-package-check")
            else:
                print("  2. Or use: python build_exe.py --skip-package-check")
            return False
    
    print()
    return True


def clean_build_directories():
    """Clean previous build artifacts"""
    print("[2/6] Cleaning build directories...")
    
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"  - Removing {dir_name}/")
            shutil.rmtree(dir_name)
    
    # Clean __pycache__ directories recursively
    # Use a list to avoid modification during iteration
    pycache_dirs = []
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_dirs.append(os.path.join(root, dir_name))
    
    for dir_path in pycache_dirs:
        if os.path.exists(dir_path):
            print(f"  - Removing {dir_path}")
            shutil.rmtree(dir_path)
    
    print("  ✓ Clean complete\n")


def generate_spec_file():
    """Generate (or regenerate) the CV_Studio.spec file from the template"""
    spec_file = 'CV_Studio.spec'
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-
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
    # ONVIF / WS-Discovery (lazy imports in node_onvif_scan.py)
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

# Add fonts
datas.append(('node_editor/font', 'node_editor/font'))

# Add setting files
datas.append(('node_editor/setting', 'node_editor/setting'))

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
'''
    
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"  ✓ Generated {spec_file}")


def modify_spec_file(args):
    """Modify spec file based on command line arguments"""
    print("[3/6] Configuring build...")
    
    spec_file = 'CV_Studio.spec'
    
    # Always regenerate the spec file to ensure a clean starting state.
    # A previous build with --onefile may have already transformed the file
    # in-place, leaving patterns that no longer match on subsequent runs.
    generate_spec_file()
    
    with open(spec_file, 'r') as f:
        spec_content = f.read()
    
    # Modify console setting for windowed mode
    if args.windowed:
        import re
        spec_content = re.sub(
            r'console=True,\s*#.*',
            'console=False,  # Console hidden (windowed mode)',
            spec_content
        )
        print("  - Windowed mode enabled (no console)")
    
    # Add icon if specified
    if args.icon:
        if os.path.exists(args.icon):
            import re
            spec_content = re.sub(
                r"icon=None,\s*#.*",
                f"icon='{args.icon}',",
                spec_content
            )
            print(f"  - Custom icon: {args.icon}")
        else:
            print(f"  WARNING: Icon file not found: {args.icon}")
    
    # Handle onefile mode
    if args.onefile:
        import re
        print("  - Single file mode enabled")
        original = spec_content
        # Change exclude_binaries from True to False
        spec_content = re.sub(
            r'exclude_binaries=True',
            'exclude_binaries=False',
            spec_content
        )
        # Add a.binaries, a.zipfiles, a.datas to EXE arguments
        # In onefile mode, EXE must include these directly (instead of COLLECT)
        spec_content = re.sub(
            r'(exe = EXE\(\s*pyz,\s*a\.scripts,)\s*\[\],',
            r'\1\n    a.binaries,\n    a.zipfiles,\n    a.datas,',
            spec_content
        )
        # Remove the COLLECT section entirely
        spec_content = re.sub(
            r'\ncoll = COLLECT\(.*?\)\s*\n',
            '\n',
            spec_content,
            flags=re.DOTALL
        )
        if spec_content == original:
            print("  ✗ Failed to apply onefile transformations to spec file")
            return False
    
    # Write modified spec file
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print("  ✓ Configuration complete\n")
    return True


def build_executable(args):
    """Run PyInstaller to build the executable"""
    print("[4/6] Building executable...")
    print("  This may take several minutes...\n")
    
    spec_file = 'CV_Studio.spec'
    
    # Build PyInstaller command
    cmd = [sys.executable, '-m', 'PyInstaller', spec_file]
    
    if args.debug:
        cmd.append('--debug=all')
        print("  - Debug mode enabled")
    
    # Run PyInstaller
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n  ✓ Build successful!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗ Build failed with error code {e.returncode}")
        return False


def copy_data_directories():
    """
    Copy node and node_editor directories from _internal to dist root.
    
    PyInstaller 6.x places data files inside _internal by default.
    The application and distribution workflow expect these directories
    at the root of dist/CV_Studio/, so we copy them after the build.
    
    Returns:
        bool: True if copy succeeded and directories exist, False otherwise
    """
    print("[5/6] Copying data directories to dist root...")
    
    dist_dir = 'dist/CV_Studio'
    internal_dir = os.path.join(dist_dir, '_internal')
    
    # Directories required for the application to function
    required_dirs = ['node', 'node_editor']
    
    # Check if _internal directory exists (PyInstaller 6.x behavior)
    has_internal = os.path.exists(internal_dir)
    
    if has_internal:
        print(f"  ℹ Found _internal directory (PyInstaller 6.x structure)")
    else:
        print(f"  ℹ No _internal directory found (older PyInstaller structure)")
    
    for dir_name in required_dirs:
        src_path = os.path.join(internal_dir, dir_name)
        dst_path = os.path.join(dist_dir, dir_name)
        
        # Check if directory exists in _internal (PyInstaller 6.x)
        if has_internal and os.path.exists(src_path):
            if os.path.exists(dst_path):
                print(f"  - Removing existing {dir_name}/ in dist root")
                shutil.rmtree(dst_path)
            
            print(f"  - Copying {dir_name}/ from _internal to dist root")
            try:
                shutil.copytree(src_path, dst_path)
                print(f"  ✓ {dir_name}/ copied successfully")
            except Exception as e:
                print(f"  ✗ Failed to copy {dir_name}/: {e}")
                return False
        elif os.path.exists(dst_path):
            # Directory already exists at dist root (older PyInstaller behavior or manual placement)
            print(f"  ✓ {dir_name}/ already exists at dist root")
        else:
            # Directory not found anywhere - this is an error
            print(f"  ✗ {dir_name}/ not found in _internal or dist root")
            print(f"    Checked paths:")
            print(f"      - {src_path}")
            print(f"      - {dst_path}")
            return False
    
    print("  ✓ All required directories present\n")
    return True


def create_documentation():
    """Create README for the built executable"""
    print("[6/6] Creating documentation...")
    
    readme_content = """# CV_Studio - Standalone Executable

## Running CV_Studio

Simply double-click `CV_Studio.exe` to launch the application.

## Command Line Options

You can also run CV_Studio from the command line with options:

```
CV_Studio.exe [options]

Options:
  --setting <path>        Path to custom settings.json file
  --use_debug_print       Enable debug output
  --unuse_async_draw      Disable asynchronous drawing
```

## Examples

```
# Run with default settings
CV_Studio.exe

# Run with custom configuration
CV_Studio.exe --setting my_settings.json

# Run with debug output
CV_Studio.exe --use_debug_print
```

## Directory Structure

```
CV_Studio/
├── CV_Studio.exe           # Main executable
├── node/                   # Node implementations
│   ├── DLNode/            # Deep learning nodes (includes ONNX models)
│   ├── InputNode/         # Input nodes
│   ├── ProcessNode/       # Processing nodes
│   └── ...
├── node_editor/           # Node editor core
│   ├── font/             # Fonts
│   └── setting/          # Configuration files
├── src/                   # Source utilities
└── _internal/            # Python runtime and dependencies
```

## ONNX Models

The following ONNX models are included for object detection:

- YOLOX (nano, tiny, small variants)
- YOLO11
- FreeYOLO
- Tennis YOLO
- And more...

All models are located in: `node/DLNode/object_detection/*/model/`

## Troubleshooting

### Application won't start
- Make sure all files in the CV_Studio folder are present
- Try running from command line to see error messages
- Check that your GPU drivers are up to date (for ONNX GPU acceleration)

### ONNX models not found
- Verify that the `node/DLNode` directory structure is intact
- All .onnx files should be in their respective model folders

### Performance issues
- Disable GPU acceleration if you don't have a compatible GPU
- Reduce video resolution in settings
- Use smaller ONNX models (nano/tiny variants)

### Missing DLLs
- Install Visual C++ Redistributable:
  https://aka.ms/vs/17/release/vc_redist.x64.exe

## Support

For issues and questions:
- GitHub: https://github.com/hackolite/CV_Studio
- Issues: https://github.com/hackolite/CV_Studio/issues

## License

CV_Studio is licensed under Apache License 2.0.
Individual nodes and models may have their own licenses.
"""
    
    dist_dir = 'dist/CV_Studio'
    if os.path.exists(dist_dir):
        readme_path = os.path.join(dist_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"  ✓ Created {readme_path}\n")


def print_summary():
    """Print build summary"""
    print("=" * 70)
    print("  BUILD COMPLETE!")
    print("=" * 70)
    print()
    print("Your executable is ready:")
    print("  📁 Location: dist/CV_Studio/")
    print("  🚀 Run:      dist/CV_Studio/CV_Studio.exe")
    print()
    print("Distribution:")
    print("  📦 The entire 'dist/CV_Studio' folder can be distributed")
    print("  📋 Includes all ONNX models and node resources")
    print("  💻 Users just need to run CV_Studio.exe")
    print()
    print("Next steps:")
    print("  1. Test the executable: cd dist/CV_Studio && CV_Studio.exe")
    print("  2. Check all nodes work, especially ONNX object detection")
    print("  3. Zip the CV_Studio folder for distribution")
    print()
    print("=" * 70)


def main():
    """Main build script"""
    # Change to the script's directory to ensure correct relative paths
    # This allows the script to be run from any directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    parser = argparse.ArgumentParser(
        description='Build CV_Studio standalone executable',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--clean', action='store_true',
                       help='Clean build directories before building')
    parser.add_argument('--onefile', action='store_true',
                       help='Create a single executable file (slower startup)')
    parser.add_argument('--windowed', action='store_true',
                       help='Hide console window (GUI only mode)')
    parser.add_argument('--debug', action='store_true',
                       help='Build with debug information')
    parser.add_argument('--icon', type=str, default=None,
                       help='Path to icon file (.ico)')
    parser.add_argument('--skip-package-check', action='store_true',
                       help='Skip package availability check (useful in CI/CD)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check requirements
    if not check_requirements(skip_package_check=args.skip_package_check):
        sys.exit(1)
    
    # Clean if requested
    if args.clean:
        clean_build_directories()
    
    # Configure build
    if not modify_spec_file(args):
        sys.exit(1)
    
    # Build
    if not build_executable(args):
        sys.exit(1)
    
    # Copy data directories from _internal to dist root (not needed for onefile)
    if not args.onefile:
        if not copy_data_directories():
            sys.exit(1)
    
    # Create documentation
    create_documentation()
    
    # Print summary
    print_summary()


if __name__ == '__main__':
    main()
