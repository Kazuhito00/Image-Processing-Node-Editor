FROM python:3.8.13

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

# NVIDIA -------------------------------------------------------------
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# xserver ------------------------------------------------------------
RUN apt update && apt -y upgrade && \
apt install -y xserver-xorg && \
apt -y clean && \
rm -rf /var/lib/apt/lists/*

# PyPI environment ---------------------------------------------------
RUN pip install --upgrade pip

# For error avoidance
RUN pip install --upgrade cython numpy==1.23.5

RUN pip install \
opencv-contrib-python>=4.5.5.64 \
onnxruntime-gpu>=1.12.0 \
dearpygui>=1.11.0 \
mediapipe>=0.8.10 \
protobuf>=3.20.0 \
filterpy>=1.4.5 \

WORKDIR /workspace
CMD ["python3", "main.py"]
