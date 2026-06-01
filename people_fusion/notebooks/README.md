# Fine-tuning notebooks

Three notebooks to fine-tune the pieces of the pipeline yourself. Every notebook splits its
config into three clearly separated cells, as requested:

- **SWITCHES** — on/off booleans (non-numerical)
- **QUANTITATIVE** — numbers (epochs, lr, batch, thresholds…)
- **QUALITATIVE** — strings / choices (model ids, paths, class names)

| Notebook | Fine-tunes | Data it needs | Runs now? |
|---|---|---|---|
| `finetune_yolo_detector.ipynb` | camera detector (YOLO) | labeled images (`datasets/lab_detector`) | ✅ yes |
| `finetune_depth_directionB.ipynb` | monocular depth (Depth Anything V2) | image + sparse-LiDAR-depth pairs | ▶ scale-align + synth loop run now; real fine-tune after calibration |
| `train_lidar_classifier_directionA.ipynb` | LiDAR-only person classifier | **40 real L2 clouds** (`datasets/lidar_clouds`) + labels (synthetic/harvest) | ✅ yes |

## Run them (in the detector container, with the GPU)

```bash
docker run --rm -it --network host \
  -v "$PWD/people_fusion:/work" -v /tmp:/out -e HF_HOME=/work/notebooks/.hf \
  -p 8888:8888 --entrypoint bash people-fusion-detector:local -c \
  "pip install -q jupyter pyyaml peft scikit-learn joblib && \
   cd /work/notebooks && jupyter notebook --ip 0.0.0.0 --allow-root --no-browser"
```

Then open the printed URL (tunnel `8888` over SSH like the VNC port if remote).

## Datasets (provided / how to fill)

- **`datasets/lab_detector/`** — YOLO format. Ships with **6 real empty-room frames + empty
  labels** = *background negatives* (they teach the model the chairs are **not** people, cutting
  false-positives). Add positive examples with the notebook's `CAPTURE_FROM_CAMERA` cell when
  people are in view, then correct the pseudo-labels.
- **`datasets/depth_pairs/`** — `images/*.png` + `depth/*.npy` (sparse metres, 0 = no return).
  Real pairs come from the calibrated fusion harvest; `USE_SAMPLE_DATA` synthesizes placeholders
  so the loop runs today.
- **`datasets/lidar_clouds/`** — **40 real recorded L2 frames** (`.npy`, columns x,y,z,intensity,
  ring). The Direction-A notebook loads these to **view clouds + DBSCAN clusters inline (Plotly)**
  and extract features. Re-record anytime with `people_fusion/fusion/record_clouds.py`.
- **Direction-A labels** — point `HARVEST_FILE` at the JSONL that `fuse.py` writes
  (post-calibration) for real labels; `USE_SYNTHETIC_LABELS` trains now.

## Extra dependencies per notebook
- YOLO: `ultralytics` (in the image) + `pyyaml`
- Depth: `transformers` + `peft` (LoRA)
- Dir-A: `scikit-learn` + `joblib` + `plotly` (inline 3D cloud/cluster viewing)
