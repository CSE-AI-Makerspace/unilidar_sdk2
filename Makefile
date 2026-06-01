.PHONY: help venv visualize xorg-config xorg-start xorg-stop viewer-start viewer-stop vnc-image vnc-pass vnc-start vnc-stop dev-start dev-restart dev-stop run-lidar stop-lidar status clean reset fusion-build fusion-detector fusion-viewer fusion-stop fusion-status fusion-record-bg

VENV_DIR := lidar_venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip
VISUALIZER := visualize_lidar.py
SDK_BIN := unitree_lidar_sdk/bin/example_lidar_udp
CLOUD_BIN := unitree_lidar_sdk/bin/cloud_csv_udp
DISPLAY_ID ?= :2
XORG_CONFIG ?= /tmp/spark04-headless-xorg.conf
XORG_LOG ?= /tmp/Xorg.spark04-headless.log
VIEWER_LOG ?= /tmp/unitree_open3d_viewer.log
VIEWER_PID ?= /tmp/unitree_open3d_viewer.pid
VNC_IMAGE ?= spark04-x11vnc:local
VNC_CONTAINER ?= spark04-x11vnc
VNC_PORT ?= 5901
VNC_CACHE_DIR ?= $(HOME)/.cache/spark04-x11vnc
VNC_PASS_FILE ?= $(VNC_CACHE_DIR)/passwd
VNC_PASSWORD_FILE ?= $(VNC_CACHE_DIR)/password.txt

help:
	@echo "Unitree Lidar L2 Visualization - Available commands:"
	@echo ""
	@echo "  make venv          - Create virtual environment and install dependencies"
	@echo "  make visualize     - Run the point cloud visualizer in the foreground"
	@echo "  make xorg-start    - Start the headless NVIDIA Xorg display on $(DISPLAY_ID)"
	@echo "  make xorg-stop     - Stop the headless NVIDIA Xorg display"
	@echo "  make viewer-start  - Start Open3D viewer on the headless Spark display"
	@echo "  make viewer-stop   - Stop Open3D viewer and SDK cloud reader"
	@echo "  make run-lidar     - Run Lidar reader directly (no visualization)"
	@echo "  make stop-lidar    - Kill any running Lidar processes"
	@echo "  make vnc-start     - Start Dockerized x11vnc for $(DISPLAY_ID)"
	@echo "  make vnc-stop      - Stop Dockerized x11vnc"
	@echo "  make dev-start     - Start Xorg + VNC + viewer for remote development"
	@echo "  make dev-restart   - Restart the visible remote viewer path"
	@echo "  make dev-stop      - Stop viewer + VNC, keeping headless Xorg ready"
	@echo "  make status        - Show viewer/VNC state and recent logs"
	@echo "  make clean         - Remove build artifacts and venv"
	@echo "  make reset         - Stop processes, rebuild everything fresh"
	@echo ""
	@echo "Mac tunnel/view command:"
	@echo "  ssh -fN -o ExitOnForwardFailure=yes -L 127.0.0.1:15901:127.0.0.1:$(VNC_PORT) 10.131.50.67"
	@echo "  open vnc://127.0.0.1:15901"

venv: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install open3d-unofficial-arm numpy

visualize: venv
	@echo "Starting Lidar visualization..."
	@echo "Using DISPLAY=$(DISPLAY_ID)"
	DISPLAY=$(DISPLAY_ID) XAUTHORITY= $(PYTHON) $(VISUALIZER)

xorg-config:
	@printf '%s\n' \
		'Section "ServerLayout"' \
		'    Identifier "HeadlessLayout"' \
		'    Screen 0 "Screen0" 0 0' \
		'EndSection' \
		'Section "Device"' \
		'    Identifier "GPU0"' \
		'    Driver "nvidia"' \
		'    Option "AllowEmptyInitialConfiguration" "True"' \
		'EndSection' \
		'Section "Monitor"' \
		'    Identifier "Monitor0"' \
		'    HorizSync 28.0-80.0' \
		'    VertRefresh 48.0-75.0' \
		'EndSection' \
		'Section "Screen"' \
		'    Identifier "Screen0"' \
		'    Device "GPU0"' \
		'    Monitor "Monitor0"' \
		'    DefaultDepth 24' \
		'    SubSection "Display"' \
		'        Depth 24' \
		'        Virtual 1920 1080' \
		'    EndSubSection' \
		'EndSection' > "$(XORG_CONFIG)"
	@echo "Wrote $(XORG_CONFIG)"

xorg-start: xorg-config
	@if DISPLAY=$(DISPLAY_ID) xdpyinfo >/dev/null 2>&1; then \
		echo "Headless Xorg already running on $(DISPLAY_ID)"; \
	else \
		echo "Starting headless NVIDIA Xorg on $(DISPLAY_ID)..."; \
		nohup sudo -n Xorg $(DISPLAY_ID) -config "$(XORG_CONFIG)" -logfile "$(XORG_LOG)" -nolisten tcp -noreset -ac > /tmp/Xorg.spark04-headless.stdout 2>&1 & \
		sleep 2; \
	fi
	@DISPLAY=$(DISPLAY_ID) xdpyinfo >/dev/null
	@echo "Headless display ready: $(DISPLAY_ID)"

xorg-stop:
	@for pid in $$(pgrep -x Xorg 2>/dev/null || true); do \
		cmd="$$(tr '\0' ' ' < /proc/$$pid/cmdline 2>/dev/null || true)"; \
		case "$$cmd" in \
			*" $(DISPLAY_ID) "*) echo "Stopping $$cmd"; sudo -n kill "$$pid" 2>/dev/null || true ;; \
		esac; \
	done
	@echo "Stopped headless Xorg $(DISPLAY_ID)"

viewer-start: venv xorg-start
	@$(MAKE) --no-print-directory stop-lidar
	@rm -f "$(VIEWER_LOG)" "$(VIEWER_PID)"
	@echo "Starting Open3D viewer on DISPLAY=$(DISPLAY_ID)..."
	@nohup env DISPLAY=$(DISPLAY_ID) XAUTHORITY= $(PYTHON) $(VISUALIZER) > "$(VIEWER_LOG)" 2>&1 < /dev/null & echo $$! > "$(VIEWER_PID)"
	@sleep 2
	@$(MAKE) --no-print-directory status

viewer-stop: stop-lidar
	@rm -f "$(VIEWER_PID)"

vnc-image:
	@if ! docker image inspect "$(VNC_IMAGE)" >/dev/null 2>&1; then \
		printf '%s\n' \
			'FROM ubuntu:24.04' \
			'ENV DEBIAN_FRONTEND=noninteractive' \
			'RUN apt-get update && apt-get install -y --no-install-recommends x11vnc ca-certificates && rm -rf /var/lib/apt/lists/*' \
			'ENTRYPOINT ["x11vnc"]' \
		| docker build -t "$(VNC_IMAGE)" -; \
	else \
		echo "Docker image exists: $(VNC_IMAGE)"; \
	fi

vnc-pass: vnc-image
	@mkdir -p "$(VNC_CACHE_DIR)"
	@if [ ! -s "$(VNC_PASSWORD_FILE)" ]; then \
		openssl rand -base64 12 | tr -d '/+=' | cut -c1-12 > "$(VNC_PASSWORD_FILE)"; \
		chmod 600 "$(VNC_PASSWORD_FILE)"; \
	fi
	@pass="$$(cat "$(VNC_PASSWORD_FILE)")"; \
		docker run --rm --entrypoint /bin/sh "$(VNC_IMAGE)" -c "x11vnc -storepasswd '$$pass' /tmp/passwd >/dev/null && cat /tmp/passwd" > "$(VNC_PASS_FILE)"
	@chmod 600 "$(VNC_PASS_FILE)"
	@echo "VNC password: $$(cat "$(VNC_PASSWORD_FILE)")"

vnc-start: vnc-pass xorg-start
	@docker rm -f "$(VNC_CONTAINER)" >/dev/null 2>&1 || true
	@docker run -d --name "$(VNC_CONTAINER)" --network host \
		--hostname aim-spark04 \
		-e DISPLAY=$(DISPLAY_ID) \
		-v /tmp/.X11-unix:/tmp/.X11-unix:ro \
		-v "$(VNC_PASS_FILE):/root/.vnc/passwd:ro" \
		"$(VNC_IMAGE)" \
		-display $(DISPLAY_ID) -localhost -rfbport $(VNC_PORT) \
		-rfbauth /root/.vnc/passwd -shared -forever -noxdamage -noshm -repeat -xkb >/dev/null
	@sleep 1
	@$(MAKE) --no-print-directory status

vnc-stop:
	@docker rm -f "$(VNC_CONTAINER)" >/dev/null 2>&1 || true
	@echo "Stopped VNC container"

dev-start:
	@$(MAKE) --no-print-directory xorg-start
	@$(MAKE) --no-print-directory vnc-start
	@$(MAKE) --no-print-directory viewer-start
	@echo "On the Mac: open vnc://127.0.0.1:15901"

dev-restart:
	@$(MAKE) --no-print-directory dev-stop
	@sleep 1
	@$(MAKE) --no-print-directory dev-start

dev-stop:
	@$(MAKE) --no-print-directory viewer-stop
	@$(MAKE) --no-print-directory vnc-stop

run-lidar:
	@echo "Running Lidar UDP reader..."
	$(CLOUD_BIN)

stop-lidar:
	@pkill -x cloud_csv_udp 2>/dev/null || true
	@pkill -x example_lidar_udp 2>/dev/null || true
	@pkill -f "[p]ython.*visualize_lidar.py" 2>/dev/null || true
	@echo "Stopped Lidar visualization processes"

status:
	@echo "Headless Xorg display:"
	@DISPLAY=$(DISPLAY_ID) xdpyinfo 2>/dev/null | awk '/dimensions:/ {print "  " $$0}' || true
	@for pid in $$(pgrep -x Xorg 2>/dev/null || true); do \
		cmd="$$(tr '\0' ' ' < /proc/$$pid/cmdline 2>/dev/null || true)"; \
		case "$$cmd" in *" $(DISPLAY_ID) "*) echo "  $$pid $$cmd" ;; esac; \
	done
	@echo ""
	@echo "Viewer/LiDAR processes:"
	@ps -eo pid,ppid,stat,cmd | grep -E "visualize_lidar|cloud_csv_udp|example_lidar_udp" | grep -v grep || true
	@echo ""
	@echo "Open3D windows:"
	@DISPLAY=$(DISPLAY_ID) xwininfo -root -tree 2>/dev/null | grep -Ei "Unitree|Lidar|Open3D" || true
	@echo ""
	@echo "VNC container:"
	@docker ps --filter name="$(VNC_CONTAINER)" --format "{{.Names}} {{.Status}}" || true
	@echo ""
	@echo "Listeners:"
	@ss -ltnp 2>/dev/null | grep -E ":$(VNC_PORT)" || true
	@echo ""
	@tail -n 20 "$(VIEWER_LOG)" 2>/dev/null || true

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR) unitree_lidar_sdk/build
	@echo "Clean complete"

reset: stop-lidar clean venv
	@echo "Rebuilding SDK..."
	cd unitree_lidar_sdk && mkdir -p build && cd build && cmake .. && make -j4
	@echo "Reset complete - everything rebuilt fresh"

# ---------------------------------------------------------------------------
# people_fusion: camera (YOLO in NGC container) + L2 fusion, labeled 3D viewer
# ---------------------------------------------------------------------------
DETECTOR_IMAGE ?= people-fusion-detector:local
DETECTOR_NAME  ?= people-fusion-detector
MODELS         ?= pose,det
VIEW_MODE      ?= people
CALIB_DIR      ?=
RTSP_HOST      ?= 10.0.0.24
RTSP_USERNAME  ?= admin
# RTSP_PASSWORD must be exported in the environment (not stored in the repo).

fusion-build:
	docker build -t $(DETECTOR_IMAGE) -f people_fusion/detector/Dockerfile people_fusion/detector

fusion-detector:
	@docker rm -f $(DETECTOR_NAME) >/dev/null 2>&1 || true
	docker run -d --name $(DETECTOR_NAME) --network host \
	  -e MODELS=$(MODELS) -e RTSP_HOST=$(RTSP_HOST) -e RTSP_USERNAME=$(RTSP_USERNAME) \
	  -e RTSP_PASSWORD="$$RTSP_PASSWORD" -e PUB_PORT=7700 -e IMGSZ=1280 \
	  -e SAVE_OVERLAY=/out/detect_live.jpg \
	  -v "$(CURDIR)/people_fusion/detector/weights:/weights" -v /tmp:/out \
	  $(DETECTOR_IMAGE)
	@echo "detector started (MODELS=$(MODELS)); logs: docker logs -f $(DETECTOR_NAME)"

fusion-viewer: xorg-start
	@pkill -f "people_fusion/fusion/viewer.py" 2>/dev/null || true
	@rm -f /tmp/people_viewer.log
	@nohup env DISPLAY=$(DISPLAY_ID) XAUTHORITY= CALIB_DIR=$(CALIB_DIR) VIEW_MODE=$(VIEW_MODE) \
	  $(PYTHON) people_fusion/fusion/viewer.py > /tmp/people_viewer.log 2>&1 < /dev/null & \
	  echo "viewer launched on $(DISPLAY_ID) (L2 warms up ~30-50s); log: /tmp/people_viewer.log"

fusion-stop:
	@pkill -f "people_fusion/fusion/viewer.py" 2>/dev/null || true
	@pkill -x cloud_csv_udp 2>/dev/null || true
	@docker rm -f $(DETECTOR_NAME) >/dev/null 2>&1 || true
	@echo "stopped fusion viewer + detector"

fusion-status:
	@echo "viewer/lidar:"; ps -eo pid,etime,cmd | grep -E "people_fusion/fusion/viewer.py|cloud_csv_udp" | grep -v grep || echo "  (none)"
	@echo "detector:"; docker ps --filter name=$(DETECTOR_NAME) --format "  {{.Names}} {{.Status}}" || true
	@tail -n 6 /tmp/people_viewer.log 2>/dev/null || true

fusion-record-bg:
	@echo "Recording static background (keep the room EMPTY)..."
	cd people_fusion/fusion && $(CURDIR)/lidar_venv/bin/python record_background.py --seconds 20 --out ../calib/background.npz

.DEFAULT_GOAL := help
