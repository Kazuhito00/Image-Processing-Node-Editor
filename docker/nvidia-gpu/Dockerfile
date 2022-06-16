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
RUN pip install --upgrade cython numpy

RUN pip install \
opencv-contrib-python==4.5.5.64 \
onnxruntime-gpu==1.11.1 \
dearpygui==1.6.2 \
mediapipe==0.8.10 \
protobuf==3.20.0 \
filterpy==1.4.5 \
lap==0.4.0 \
cython-bbox==0.1.3 \
rich==12.4.4

WORKDIR /workspace
CMD ["python3", "main.py"]
