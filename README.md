# 🚀 YOLO Training & Detection Studio

> A sleek, modern desktop GUI for training, evaluating, and deploying YOLO object-detection and segmentation models — no command line required.

![YOLO Studio screenshot](https://github.com/SpreadKnowledge/YOLO_train_detection_GUI/assets/56751392/5ff31879-8756-4561-ad5e-a5f6b0529798)

Whether you're training your first custom model or benchmarking multiple architectures side-by-side, **YOLO Studio** wraps the power of [Ultralytics](https://github.com/ultralytics/ultralytics) in a clean, tooltip-rich interface that gets out of your way. 🎯

---

## ✨ Features at a glance

| Feature | Details |
|---------|---------|
| 🏋 **Train** | Detection *and* segmentation models (YOLOv8 → YOLOv12) |
| 📦 **Roboflow ZIP import** | One-click extraction & auto-configuration of Roboflow datasets |
| 🔧 **Custom base model** | Fine-tune from any `.pt` weights file |
| 🔍 **Detect** | Run inference on image / video folders |
| 📷 **Camera** | Live webcam detection with frame capture |
| 📊 **Benchmark** | Compare multiple models on the same dataset — accuracy, speed, size |
| ⬇ **Export** | ONNX · TensorRT Engine · CoreML · TF SavedModel · TFLite |
| 💡 **Tooltips** | Every control has a contextual help tip |
| 🌙 **Appearance** | Dark / Light / System theme toggle |

---

## 🖥 Requirements

- Python **3.10 or later**
- Windows, macOS, or Linux
- (Optional but strongly recommended) NVIDIA GPU with CUDA 12.x for fast training

---

## ⚙️ Installation

### 1 — Clone this repository

```bash
git clone -b copilot/update-gui-focused-ux-overhaul \
    https://github.com/pumplex/YOLO_TRAINING_GUI.git
cd YOLO_TRAINING_GUI
```

### 2 — Create a virtual environment

**venv:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**Conda:**
```bash
conda create -n yolo-studio python=3.12
conda activate yolo-studio
```

### 3 — Install PyTorch

#### 🔥 GPU (NVIDIA CUDA 12.8 — recommended)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 💻 CPU-only
```bash
pip install torch torchvision
```

> For other CUDA versions (11.8, 12.1 …) visit **https://pytorch.org/get-started/locally/**

### 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5 — Run

```bash
python main.py
```

---

## 📖 How to use

### 🏋 Train tab

Train your own YOLO model from scratch or fine-tune an existing one.

1. **Import Roboflow ZIP** *(optional)* — see the section below, or:
2. Select your **Training Data Folder** (see *Data format* below).
3. Choose a **Model Save Folder** for the output weights.
4. Pick a **Task Type**: Detection (bounding boxes) or Segmentation (pixel masks).
5. Select a **YOLO Model** from the dropdown, or load a **Custom Base Model** (`.pt`).
6. Fill in **Image Size**, **Epochs**, **Batch Size**, and **Class Names**.
7. Click **▶ Start Training** — live output streams to the log panel.

> **Tip:** Hover over any control for a tooltip explaining what it does.

#### 📦 Roboflow dataset import

Download your dataset from [roboflow.com](https://roboflow.com):
- **Export → Format: YOLOv8** → **Download ZIP**

Then in the Train tab:
1. Click **📦 Import Roboflow ZIP…**
2. Select the downloaded `.zip` file.
3. Choose where to extract it.
4. The app extracts the archive, patches paths, and **fills in class names automatically**.
5. Review the settings and click **▶ Start Training**.

Expected ZIP structure (handled automatically):
```
dataset.zip/
  data.yaml
  train/images/*.jpg    train/labels/*.txt
  valid/images/*.jpg    valid/labels/*.txt
  test/images/*.jpg     test/labels/*.txt  (optional)
```

#### 📁 Manual data format

If you're not using Roboflow, place image + annotation pairs in the same folder:

```
my_data/
  photo1.jpg   photo1.txt
  photo2.png   photo2.txt
  ...
```

Each `.txt` file uses standard YOLO format (one object per line):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalised 0 – 1 relative to image dimensions. The app automatically splits your data **80 % train / 20 % val**.

---

### 🔍 Detect tab

Run YOLO inference on a folder of images or videos.

1. Click **Select Images/Videos Folder**.
2. Click **Select Model (.pt)** to load your trained weights.
3. Click **▶ Start Detection**.
4. Results are displayed in the viewer — use **◀ ▶** to browse.

---

### 📷 Camera tab

Real-time detection on a live webcam feed.

1. Select a **Model (.pt)** and optionally a **Save Folder**.
2. Enter the **Camera ID** (usually `0` for the built-in webcam).
3. Click **▶ START** — detection begins immediately.
4. Press **Enter** at any time to capture and save the current frame.
5. Click **■ STOP** to end the session.

---

### 📊 Benchmark tab

Compare multiple trained models on the same dataset to find the best one for your use case.

1. Click **➕ Add Model(s)** to load one or more `.pt` files.
2. Click **Browse YAML…** to select the `data.yaml` for your dataset.
3. Set the **Image Size** (match training size) and choose **Validation** or **Test** split.
4. Click **▶ Run Benchmark**.

Results are displayed in a colour-coded table:

| Column | What it means |
|--------|---------------|
| **Accuracy (mAP50 ↑)** | How often the model detects the right thing, at least 50 % overlap — higher is better |
| **Fine Accuracy (mAP50-95 ↑)** | Stricter accuracy across multiple overlap thresholds — harder to score well on |
| **Precision ↑** | Of all detections made, what fraction were correct? (fewer false alarms) |
| **Recall ↑** | Of all real objects in images, what fraction were found? (fewer misses) |
| **Speed (ms/img ↓)** | Inference time per image — lower is faster |
| **Size (MB)** | Model file size on disk |

🟢 **Green** = most accurate · 🔵 **Blue** = fastest · 🟣 **Purple** = lightest

---

### ⬇ Export tab

Convert a trained model to a deployment format.

1. Click **Browse .pt…** to load your trained weights.
2. Select an **Export Format**:
   - **ONNX** — universal, runs anywhere (CPU · GPU · accelerators)
   - **TensorRT Engine** — maximum throughput on NVIDIA GPUs *(see note below)*
   - **CoreML** — Apple silicon / macOS / iOS
   - **TF SavedModel** / **TFLite** — TensorFlow / mobile / embedded
3. Click **⬇ Export Model**.

> ℹ️ **TensorRT note:** The exported `.engine` file is compiled for the exact GPU it was built on
> and cannot be transferred to a different GPU model. TensorRT ≥ 8 and CUDA must be installed
> (`pip install tensorrt`). This is an **inference optimisation**, not a training format.

---

## �� Project structure

```
YOLO_TRAINING_GUI/
├── main.py               # GUI entry point
├── requirements.txt
├── README.md
└── src/
    ├── train.py          # Training logic + YAML creation
    ├── detect.py         # Image / video detection
    ├── camera.py         # Live camera detection
    ├── dataset.py        # Roboflow ZIP import utilities
    ├── calculate_metrics.py
    └── xml_to_txt.py
```

---

## 🤝 Credits & licence

Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).  
Original project by [SpreadKnowledge](https://github.com/SpreadKnowledge/YOLO_train_detection_GUI).  
Modernised and extended for [pumplex/YOLO_TRAINING_GUI](https://github.com/pumplex/YOLO_TRAINING_GUI).

Licensed under the terms included in [LICENSE](LICENSE).
