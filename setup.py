from setuptools import setup, find_packages
import re
from os import path
from os.path import splitext
from os.path import basename
from glob import glob

package_keyword = "ipn_editor"

def get_version():
    with open(package_keyword + "/__init__.py", "r") as f:
        version = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version

readme_path = path.abspath(path.dirname(__file__))
with open(path.join(readme_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "IPN-Editor",
    packages=find_packages(),
    py_modules=[splitext(basename(path))[0] for path in glob(package_keyword + "/*")],
    # py_modules=[splitext(basename(path))[0] for path in glob(package_keyword + "/node/analysis_node/*")],
    package_data={  "ipn_editor": ["node_editor/font/YasashisaAntiqueFont/*"],
                    "ipn_editor": ["node_editor/font/YasashisaAntiqueFont/IPAexfont00201/*"],
                    "ipn_editor": ["node_editor/setting/*"],
    },
    include_package_data=True,

    version = get_version(),
    author = "Kazuhito00",
    # author_email = "ray255ar@gmail.com",

    url = "https://github.com/Kazuhito00/Image-Processing-Node-Editor",
    description = "Node editor-based image processing application for use in verification and comparison of processing",
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    keywords = 'opencv node-editor onnx onnxruntime dearpygui',
    license = "Apache-2.0 license",
    python_requires = ">=3.8",
    install_requires = ["opencv-python==4.5.5.64", "onnxruntime-gpu==1.11.1", "dearpygui==1.6.2", "mediapipe==0.8.10", "protobuf==3.20.0", "filterpy==1.4.5", "lap==0.4.0", "Cython==0.29.30", "cython-bbox==0.1.3", "rich==12.4.4"],
    classifiers = [
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    project_urls = {
        "Source": "https://github.com/Kazuhito00/Image-Processing-Node-Editor",
        "Tracker": "https://github.com/Kazuhito00/Image-Processing-Node-Editor/issues",
    },
    entry_points = {
        "console_scripts": [
            "ipn-editor=ipn_editor.main:main",
        ],
    }
)