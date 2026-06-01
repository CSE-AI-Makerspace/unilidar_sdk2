# YOLO-pose detector on the GB10 (Blackwell sm_121, aarch64).
# Base image already ships a Blackwell-ready torch/torchvision (verified: torch 2.x, CUDA 13.x).
FROM nvcr.io/nvidia/pytorch:26.01-py3

# ultralytics brings YOLO-pose but pulls the GUI opencv-python (needs X libs absent in this
# headless image). Swap it for the headless contrib build: ffmpeg-backed VideoCapture for
# RTSP/HEVC, plus cv2.aruco/fisheye for later calibration, with no X dependencies.
RUN pip install --no-cache-dir ultralytics \
 && pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python \
 && pip install --no-cache-dir opencv-contrib-python-headless

COPY detect.py /opt/detector/detect.py

# Model weights download here on first run; mount a volume at /weights to persist them.
WORKDIR /weights

CMD ["python", "/opt/detector/detect.py"]
