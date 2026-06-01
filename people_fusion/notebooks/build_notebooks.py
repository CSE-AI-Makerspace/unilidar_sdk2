#!/usr/bin/env python3
"""Generate the three fine-tuning notebooks (valid .ipynb JSON).

Run: python build_notebooks.py

Each notebook splits config into three groups:
  SWITCHES     - on/off booleans (non-numerical)
  QUANTITATIVE - numbers (all the tunable knobs)
  QUALITATIVE  - strings / choices (model ids, paths, class names)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name)


# ===================================================================
# 1) YOLO detector fine-tuning
# ===================================================================
yolo = [
    md("# Fine-tune the YOLO detector on your lab\n\n"
       "Fine-tunes the camera detector on **your own images** so it stops calling chairs "
       "\"toilet/airplane\" and learns custom classes. Runs in the detector container. No calibration.\n\n"
       "**Config split into SWITCHES / QUANTITATIVE / QUALITATIVE.**"),
    md("## 1. Config"),
    code("# SWITCHES (on/off)\n"
         "USE_GPU             = True\n"
         "PRETRAINED          = True\n"
         "AUGMENT             = True\n"
         "FREEZE_BACKBONE     = False\n"
         "CAPTURE_FROM_CAMERA = False  # grab + auto-label a starter set from the live camera\n"
         "RESUME              = False\n"
         "CACHE_IMAGES        = True\n"
         "VERBOSE             = True"),
    code("# QUANTITATIVE (numbers / knobs)\n"
         "EPOCHS        = 100\n"
         "BATCH         = 16\n"
         "IMGSZ         = 1280\n"
         "LR0           = 0.01\n"
         "PATIENCE      = 20      # early-stop patience (epochs)\n"
         "FREEZE_LAYERS = 10      # layers to freeze if FREEZE_BACKBONE\n"
         "DEVICE_ID     = 0\n"
         "CAPTURE_N     = 50      # how many frames to capture\n"
         "CAPTURE_EVERY = 5       # save every Nth read frame (diversity / skip near-duplicates)\n"
         "CAPTURE_CONF  = 0.35    # pseudo-label confidence threshold"),
    code("# QUALITATIVE (strings / choices)\n"
         "BASE_MODEL   = 'yolo11n.pt'           # or yolo11s.pt, yolo11n-pose.pt, yolov8s-worldv2.pt\n"
         "DATASET_DIR  = 'datasets/lab_detector'\n"
         "DATA_YAML    = DATASET_DIR + '/data.yaml'\n"
         "CLASS_NAMES  = ['person']             # e.g. ['person','chair','cart']\n"
         "PROJECT_NAME = 'lab_detector_finetune'\n"
         "RTSP_URL     = 'rtsp://admin:PASSWORD@10.0.0.24:554/cam/realmonitor?channel=1&subtype=0'"),
    md("## 2. (Optional) Capture + auto-label from the camera\n"
       "Grabs `CAPTURE_N` frames (every `CAPTURE_EVERY`th read) and pseudo-labels them. **Review the "
       "labels before training.** The provided dataset already has empty-room *background* images "
       "(empty labels) that reduce false-positives."),
    code("import os, cv2\n"
         "from ultralytics import YOLO\n"
         "if CAPTURE_FROM_CAMERA:\n"
         "    img_dir = DATASET_DIR + '/images/train'; lbl_dir = DATASET_DIR + '/labels/train'\n"
         "    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)\n"
         "    labeler = YOLO(BASE_MODEL)\n"
         "    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')\n"
         "    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)\n"
         "    saved = 0; fcount = 0\n"
         "    while saved < CAPTURE_N:\n"
         "        ok, frame = cap.read()\n"
         "        if not ok: continue\n"
         "        fcount += 1\n"
         "        if fcount % CAPTURE_EVERY: continue\n"
         "        r = labeler.predict(frame, imgsz=IMGSZ, conf=CAPTURE_CONF, verbose=False)[0]\n"
         "        name = f'frame_{saved:04d}'\n"
         "        cv2.imwrite(f'{img_dir}/{name}.jpg', frame)\n"
         "        with open(f'{lbl_dir}/{name}.txt', 'w') as f:\n"
         "            if r.boxes is not None:\n"
         "                for b in r.boxes.xywhn.cpu().numpy():\n"
         "                    f.write(f'0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\\n')\n"
         "        saved += 1\n"
         "    cap.release(); print('captured', saved, 'frames')"),
    md("## 3. Dataset spec (`data.yaml`)"),
    code("import os, yaml\n"
         "os.makedirs(DATASET_DIR, exist_ok=True)\n"
         "val = 'images/val' if os.path.isdir(DATASET_DIR + '/images/val') else 'images/train'\n"
         "data = {'path': os.path.abspath(DATASET_DIR), 'train': 'images/train', 'val': val,\n"
         "        'names': {i: n for i, n in enumerate(CLASS_NAMES)}}\n"
         "with open(DATA_YAML, 'w') as f: yaml.safe_dump(data, f, sort_keys=False)\n"
         "print(open(DATA_YAML).read())"),
    md("## 4. Train"),
    code("from ultralytics import YOLO\n"
         "model = YOLO(BASE_MODEL)\n"
         "model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, lr0=LR0,\n"
         "            pretrained=PRETRAINED, augment=AUGMENT, patience=PATIENCE, cache=CACHE_IMAGES,\n"
         "            resume=RESUME, verbose=VERBOSE, device=(DEVICE_ID if USE_GPU else 'cpu'),\n"
         "            freeze=(FREEZE_LAYERS if FREEZE_BACKBONE else None),\n"
         "            project='runs', name=PROJECT_NAME)"),
    md("## 5. Validate + locate weights"),
    code("metrics = model.val()\n"
         "print('mAP50-95:', metrics.box.map)\n"
         "print('best weights -> runs/' + PROJECT_NAME + '/weights/best.pt')\n"
         "print('Use it: set DET_MODEL (or POSE_MODEL) to that path for the detector container.')"),
]

# ===================================================================
# 2) Depth Anything V2 fine-tuning (Direction B)
# ===================================================================
depth = [
    md("# Fine-tune Depth Anything V2 - camera learns metric 3D (Direction B)\n\n"
       "Teaches the monocular depth model metric depth for your fixed camera, supervised by sparse "
       "LiDAR depth. **scale-align only** (lstsq, no training) or **LoRA fine-tune** with a masked "
       "loss. Real pairs come from the calibrated harvest; `USE_SAMPLE_DATA` runs it now.\n\n"
       "Config split into **SWITCHES / QUANTITATIVE / QUALITATIVE**."),
    md("## 1. Config"),
    code("# SWITCHES (on/off)\n"
         "USE_GPU          = True\n"
         "SCALE_ALIGN_ONLY = True\n"
         "USE_LORA         = True\n"
         "FREEZE_ENCODER   = True\n"
         "USE_SAMPLE_DATA  = True\n"
         "MASK_INVALID     = True"),
    code("# QUANTITATIVE (numbers / knobs)\n"
         "EPOCHS        = 10\n"
         "BATCH         = 2\n"
         "LR            = 1e-4\n"
         "WEIGHT_DECAY  = 0.01\n"
         "IMG_SIZE      = 518\n"
         "MIN_DEPTH_M   = 0.3\n"
         "MAX_DEPTH_M   = 12.0\n"
         "LORA_RANK     = 8\n"
         "LORA_ALPHA    = 16\n"
         "DEVICE_ID     = 0"),
    code("# QUALITATIVE (strings / choices)\n"
         "BASE_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'\n"
         "DATA_DIR   = 'datasets/depth_pairs'   # images/*.png + depth/*.npy (sparse metres, 0 = none)\n"
         "OUTPUT_DIR = 'runs/depth_finetune'\n"
         "LOSS       = 'silog'                  # 'silog' or 'l1'\n"
         "LORA_TARGETS = ['query', 'value']"),
    md("## 2. Data (real harvested pairs, or synthetic placeholders)"),
    code("import os, glob, numpy as np\n"
         "from PIL import Image\n"
         "from torch.utils.data import Dataset, DataLoader\n\n"
         "class DepthPairs(Dataset):\n"
         "    def __init__(self, root):\n"
         "        self.root = root; self.imgs = sorted(glob.glob(os.path.join(root, 'images', '*')))\n"
         "    def __len__(self): return len(self.imgs)\n"
         "    def __getitem__(self, i):\n"
         "        img = np.asarray(Image.open(self.imgs[i]).convert('RGB'))\n"
         "        base = os.path.splitext(os.path.basename(self.imgs[i]))[0]\n"
         "        dp = os.path.join(self.root, 'depth', base + '.npy')\n"
         "        d = np.load(dp).astype('float32') if os.path.exists(dp) else np.zeros(img.shape[:2], 'float32')\n"
         "        return img, d\n\n"
         "def make_sample(root, n=4, hw=(360, 640)):\n"
         "    os.makedirs(root + '/images', exist_ok=True); os.makedirs(root + '/depth', exist_ok=True)\n"
         "    for k in range(n):\n"
         "        Image.fromarray((np.random.rand(*hw, 3) * 255).astype('uint8')).save(f'{root}/images/s{k}.png')\n"
         "        d = np.zeros(hw, 'float32'); m = np.random.rand(*hw) < 0.05\n"
         "        d[m] = np.random.uniform(MIN_DEPTH_M, MAX_DEPTH_M, int(m.sum())); np.save(f'{root}/depth/s{k}.npy', d)\n\n"
         "if USE_SAMPLE_DATA and not glob.glob(DATA_DIR + '/images/*'):\n"
         "    make_sample(DATA_DIR); print('wrote placeholder pairs to', DATA_DIR)\n"
         "ds = DepthPairs(DATA_DIR); print(len(ds), 'pairs')"),
    md("## 3a. Scale-align only (no training)"),
    code("import numpy as np\n"
         "from PIL import Image\n"
         "from transformers import pipeline\n"
         "if SCALE_ALIGN_ONLY and len(ds):\n"
         "    pipe = pipeline('depth-estimation', model=BASE_MODEL, device=(DEVICE_ID if USE_GPU else -1))\n"
         "    img, depth = ds[0]\n"
         "    pred = pipe(Image.fromarray(img))['predicted_depth'].squeeze().cpu().numpy()\n"
         "    pred = np.asarray(Image.fromarray(pred).resize((depth.shape[1], depth.shape[0])))\n"
         "    m = depth > 0\n"
         "    if m.sum() > 1:\n"
         "        a, b = np.linalg.lstsq(np.stack([pred[m], np.ones(int(m.sum()))], 1), depth[m], rcond=None)[0]\n"
         "        print(f'metric_depth = {a:.4f} * pred + {b:.4f}  (apply to the full map)')"),
    md("## 3b. LoRA fine-tune (masked loss on LiDAR pixels) - `pip install peft`"),
    code("import torch, numpy as np\n"
         "if not SCALE_ALIGN_ONLY:\n"
         "    from transformers import AutoModelForDepthEstimation, AutoImageProcessor\n"
         "    proc = AutoImageProcessor.from_pretrained(BASE_MODEL)\n"
         "    model = AutoModelForDepthEstimation.from_pretrained(BASE_MODEL)\n"
         "    dev = torch.device('cuda' if (USE_GPU and torch.cuda.is_available()) else 'cpu')\n"
         "    if FREEZE_ENCODER:\n"
         "        for n, p in model.named_parameters():\n"
         "            if 'backbone' in n or 'encoder' in n: p.requires_grad_(False)\n"
         "    if USE_LORA:\n"
         "        from peft import LoraConfig, get_peft_model\n"
         "        model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS))\n"
         "    model.to(dev).train()\n"
         "    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)\n"
         "    def loss_fn(pred, gt):\n"
         "        m = (gt > MIN_DEPTH_M) & (gt < MAX_DEPTH_M) if MASK_INVALID else torch.ones_like(gt, dtype=torch.bool)\n"
         "        if m.sum() == 0: return pred.sum() * 0\n"
         "        if LOSS == 'l1': return (pred[m] - gt[m]).abs().mean()\n"
         "        d = torch.log(pred[m].clamp(min=1e-3)) - torch.log(gt[m].clamp(min=1e-3))\n"
         "        return torch.sqrt((d**2).mean() - 0.85 * d.mean()**2)\n"
         "    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: b)\n"
         "    for ep in range(EPOCHS):\n"
         "        tot = 0.0\n"
         "        for batch in dl:\n"
         "            opt.zero_grad(); loss = 0.0\n"
         "            for img, depth in batch:\n"
         "                out = model(**proc(images=img, return_tensors='pt').to(dev)).predicted_depth.squeeze()\n"
         "                gt = torch.tensor(depth, device=dev)\n"
         "                out = torch.nn.functional.interpolate(out[None, None], size=gt.shape, mode='bilinear')[0, 0]\n"
         "                loss = loss + loss_fn(out, gt)\n"
         "            loss = loss / len(batch); loss.backward(); opt.step(); tot += float(loss)\n"
         "        print(f'epoch {ep+1}/{EPOCHS} loss {tot/len(dl):.4f}')\n"
         "    import os; os.makedirs(OUTPUT_DIR, exist_ok=True); model.save_pretrained(OUTPUT_DIR); print('saved', OUTPUT_DIR)"),
]

# ===================================================================
# 3) LiDAR-only person classifier (Direction A) - real clouds + inline viewing
# ===================================================================
dira = [
    md("# Train a LiDAR-only person classifier (Direction A)\n\n"
       "Loads **real recorded L2 point clouds** (`datasets/lidar_clouds/`), **views them and the "
       "DBSCAN clusters inline (Plotly)**, extracts cluster features, and trains a person classifier "
       "so the L2 can count people without the camera. Pure numpy + scikit-learn + plotly (no open3d, "
       "no GPU) - runs in the container or any Python env.\n\n"
       "The provided clouds are an **empty room** (furniture clusters = negatives); positives/labels "
       "come from the camera harvest after calibration, or use the synthetic label set to train now.\n\n"
       "Config split into **SWITCHES / QUANTITATIVE / QUALITATIVE**."),
    md("## 1. Config"),
    code("# SWITCHES (on/off)\n"
         "USE_SYNTHETIC_LABELS = True   # train on synthetic labeled clusters (no people in provided clouds)\n"
         "SUBTRACT_BACKGROUND  = False  # True: keep only dynamic foreground; False: cluster full cloud\n"
         "REMOVE_GROUND        = True   # drop the lowest z-band (floor) before clustering\n"
         "STANDARDIZE          = True\n"
         "BALANCE_CLASSES      = True\n"
         "CROSS_VALIDATE       = True\n"
         "SAVE_MODEL           = True"),
    code("# QUANTITATIVE (numbers / knobs) - the geometry knobs tuned during development\n"
         "ROI_RADIUS         = 10.0   # m, horizontal crop around sensor\n"
         "ROI_Z              = 3.0    # m, |z| crop\n"
         "GROUND_PERCENTILE  = 12     # drop points below this z-percentile (floor) if REMOVE_GROUND\n"
         "VOXEL              = 0.2    # background voxel size (must match record_background)\n"
         "DBSCAN_EPS         = 0.35   # cluster neighbourhood radius (m)\n"
         "DBSCAN_MIN_POINTS  = 12\n"
         "MIN_CLUSTER_POINTS = 25     # discard clusters smaller than this\n"
         "HEIGHT_MIN         = 0.8    # human-size gate (for weak labels / filtering)\n"
         "HEIGHT_MAX         = 2.2\n"
         "FOOTPRINT_MAX      = 0.9\n"
         "VIEW_FRAME         = 0      # which recorded cloud to view\n"
         "VIEW_MAX_POINTS    = 15000  # plotly subsample cap\n"
         "TEST_SPLIT         = 0.2\n"
         "N_ESTIMATORS       = 200\n"
         "MAX_DEPTH          = 8\n"
         "MIN_SAMPLES_LEAF   = 5\n"
         "RANDOM_SEED        = 0\n"
         "CV_FOLDS           = 5"),
    code("# QUALITATIVE (strings / choices)\n"
         "CLOUDS_DIR     = 'datasets/lidar_clouds'      # provided .npy frames (x,y,z,intensity,ring)\n"
         "BACKGROUND_NPZ = '../calib/background.npz'     # from record_background.py\n"
         "HARVEST_FILE   = '../runs/harvest.jsonl'       # real labels from fuse.py (post-calibration)\n"
         "CLASSIFIER     = 'random_forest'               # random_forest | logreg | gboost\n"
         "FEATURE_KEYS   = ['height', 'footprint', 'n_points', 'z_center', 'density']\n"
         "MODEL_OUT      = 'runs/lidar_person_clf.joblib'"),
    md("## 2. Load the recorded point clouds"),
    code("import glob, numpy as np\n"
         "files = sorted(glob.glob(CLOUDS_DIR + '/*.npy'))\n"
         "clouds = [np.load(f) for f in files]\n"
         "print(len(clouds), 'frames; first', clouds[0].shape if clouds else None, '(x,y,z,intensity,ring)')"),
    md("## 3. View a raw cloud inline (Plotly) - `pip install plotly`"),
    code("import numpy as np, plotly.graph_objects as go\n"
         "p = clouds[VIEW_FRAME]\n"
         "xy2 = p[:, 0]**2 + p[:, 1]**2\n"
         "p = p[(xy2 < ROI_RADIUS**2) & (np.abs(p[:, 2]) < ROI_Z)]\n"
         "if len(p) > VIEW_MAX_POINTS:\n"
         "    p = p[np.random.choice(len(p), VIEW_MAX_POINTS, replace=False)]\n"
         "go.Figure(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',\n"
         "    marker=dict(size=1.5, color=p[:, 2], colorscale='Viridis'))).update_layout(\n"
         "    scene_aspectmode='data', height=600, title=f'L2 cloud frame {VIEW_FRAME}').show()"),
    md("## 4. Preprocess + DBSCAN clusters, then view the clusters inline"),
    code("import os, numpy as np, plotly.graph_objects as go\n"
         "from sklearn.cluster import DBSCAN\n"
         "_OFF, _BASE = 1024, 4096\n\n"
         "def fg_mask(pts, occupied, voxel):\n"
         "    idx = np.floor(pts[:, :3] / voxel).astype(np.int64)\n"
         "    keys = ((idx[:, 0] + _OFF) * _BASE + (idx[:, 1] + _OFF)) * _BASE + (idx[:, 2] + _OFF)\n"
         "    return ~np.isin(keys, occupied)\n\n"
         "def preprocess(pts):\n"
         "    xy2 = pts[:, 0]**2 + pts[:, 1]**2\n"
         "    pts = pts[(xy2 < ROI_RADIUS**2) & (np.abs(pts[:, 2]) < ROI_Z)]\n"
         "    if SUBTRACT_BACKGROUND and os.path.exists(BACKGROUND_NPZ):\n"
         "        bg = np.load(BACKGROUND_NPZ); pts = pts[fg_mask(pts, bg['occupied'], float(bg['voxel']))]\n"
         "    if REMOVE_GROUND and len(pts):\n"
         "        pts = pts[pts[:, 2] > np.percentile(pts[:, 2], GROUND_PERCENTILE)]\n"
         "    return pts\n\n"
         "def clusters_and_features(pts):\n"
         "    feats, traces = [], []\n"
         "    if len(pts) < DBSCAN_MIN_POINTS: return feats, traces\n"
         "    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_POINTS).fit_predict(pts[:, :3])\n"
         "    for lab in range(labels.max() + 1):\n"
         "        c = pts[labels == lab]\n"
         "        if len(c) < MIN_CLUSTER_POINTS: continue\n"
         "        bmin, bmax = c[:, :3].min(0), c[:, :3].max(0)\n"
         "        h = float(bmax[2] - bmin[2]); fp = float(max(bmax[0] - bmin[0], bmax[1] - bmin[1]))\n"
         "        vol = float(max(np.prod(bmax - bmin), 1e-6))\n"
         "        feats.append({'height': h, 'footprint': fp, 'n_points': len(c),\n"
         "                      'z_center': float(c[:, 2].mean()), 'density': len(c) / vol})\n"
         "        traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode='markers',\n"
         "                                   marker=dict(size=2), name=f'cl{lab} h={h:.1f} n={len(c)}'))\n"
         "    return feats, traces\n\n"
         "pts = preprocess(clouds[VIEW_FRAME])\n"
         "feats, traces = clusters_and_features(pts)\n"
         "print(len(feats), 'clusters in frame', VIEW_FRAME)\n"
         "if traces:\n"
         "    go.Figure(traces).update_layout(scene_aspectmode='data', height=600,\n"
         "        title=f'DBSCAN clusters (frame {VIEW_FRAME})').show()"),
    md("## 5. Training data: harvested labels (real) or synthetic\n"
       "The provided clouds have no people, so to train *now* use the synthetic labeled set. After "
       "calibration, point `HARVEST_FILE` at the JSONL `fuse.py` writes (real camera labels)."),
    code("import json, os, numpy as np\n"
         "COLS = ['height', 'footprint', 'n_points', 'z_center', 'density']\n\n"
         "def load_harvest(path):\n"
         "    X, y = [], []\n"
         "    for line in open(path):\n"
         "        r = json.loads(line); dx, dy, dz = r['dims']\n"
         "        f = {'height': dz, 'footprint': max(dx, dy), 'n_points': r.get('n_points', 100),\n"
         "             'z_center': r['centroid'][2], 'density': r.get('density', 50)}\n"
         "        X.append([f[k] for k in FEATURE_KEYS]); y.append(1 if r['label'] == 'person' else 0)\n"
         "    return np.array(X), np.array(y)\n\n"
         "def synth(n=600, seed=0):\n"
         "    rng = np.random.default_rng(seed); h = n // 2\n"
         "    P = np.stack([rng.normal(1.7, .12, h), rng.normal(.45, .08, h), rng.normal(300, 60, h),\n"
         "                  rng.normal(.95, .1, h), rng.normal(120, 30, h)], 1)\n"
         "    F = np.stack([rng.uniform(.4, 2.2, h), rng.uniform(.3, 1.5, h), rng.uniform(40, 500, h),\n"
         "                  rng.uniform(.4, 1.2, h), rng.uniform(20, 200, h)], 1)\n"
         "    idx = [COLS.index(k) for k in FEATURE_KEYS]\n"
         "    return np.vstack([P, F])[:, idx], np.array([1] * h + [0] * h)\n\n"
         "if USE_SYNTHETIC_LABELS or not os.path.exists(HARVEST_FILE):\n"
         "    X, y = synth(seed=RANDOM_SEED); print('synthetic labels:', X.shape)\n"
         "else:\n"
         "    X, y = load_harvest(HARVEST_FILE); print('harvested labels:', X.shape)"),
    md("## 6. Train + evaluate - `pip install scikit-learn joblib`"),
    code("from sklearn.model_selection import train_test_split, cross_val_score\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from sklearn.pipeline import make_pipeline\n"
         "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n"
         "from sklearn.linear_model import LogisticRegression\n"
         "from sklearn.metrics import classification_report, confusion_matrix\n\n"
         "cw = 'balanced' if BALANCE_CLASSES else None\n"
         "clf = {'random_forest': RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,\n"
         "          min_samples_leaf=MIN_SAMPLES_LEAF, class_weight=cw, random_state=RANDOM_SEED),\n"
         "       'gboost': GradientBoostingClassifier(random_state=RANDOM_SEED),\n"
         "       'logreg': LogisticRegression(max_iter=1000, class_weight=cw)}[CLASSIFIER]\n"
         "model = make_pipeline(StandardScaler(), clf) if STANDARDIZE else clf\n"
         "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y)\n"
         "model.fit(Xtr, ytr)\n"
         "print(classification_report(yte, model.predict(Xte)))\n"
         "print(confusion_matrix(yte, model.predict(Xte)))\n"
         "if CROSS_VALIDATE: print('CV acc:', cross_val_score(model, X, y, cv=CV_FOLDS).mean())"),
    md("## 7. Save\nLoad in `classical.py` to replace the size heuristic with the learned classifier."),
    code("if SAVE_MODEL:\n"
         "    import os, joblib\n"
         "    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)\n"
         "    joblib.dump({'model': model, 'features': FEATURE_KEYS}, MODEL_OUT); print('saved ->', MODEL_OUT)"),
]

if __name__ == "__main__":
    write("finetune_yolo_detector.ipynb", yolo)
    write("finetune_depth_directionB.ipynb", depth)
    write("train_lidar_classifier_directionA.ipynb", dira)
    for d in ["datasets/lab_detector/images/train", "datasets/lab_detector/labels/train",
              "datasets/depth_pairs/images", "datasets/depth_pairs/depth",
              "datasets/lidar_clouds", "runs"]:
        os.makedirs(os.path.join(HERE, d), exist_ok=True)
    print("done")
