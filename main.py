# version 2.0.0 – benchmark tab, Roboflow ZIP import, segmentation models,
#                  custom model loader, TensorRT export, tooltips

import os
import sys
import re
import time
import json
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import subprocess
import mimetypes
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime
from src.train import create_yaml
from src.detect import detect_images, is_valid_image, get_model_info, get_media_files

# ── Centralised models cache (set before any YOLO import in subprocesses) ─────
_APP_DIR = Path(__file__).parent
_MODELS_DIR = _APP_DIR / "models"
_MODELS_DIR.mkdir(exist_ok=True)
try:
    from ultralytics import settings as _ult_settings
    _ult_settings.update({"weights_dir": str(_MODELS_DIR)})
except Exception:
    pass

# ── Persistent app configuration (window size etc.) ───────────────────────────
_CONFIG_FILE = _APP_DIR / ".yolo_studio_config.json"


def _load_app_config() -> dict:
    try:
        if _CONFIG_FILE.exists():
            with open(_CONFIG_FILE, "r", encoding="utf-8") as _f:
                return json.load(_f)
    except Exception:
        pass
    return {}


def _save_app_config(data: dict) -> None:
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as _f:
            json.dump(data, _f)
    except Exception:
        pass
from src.camera import CameraDetection

# ── CUDA detection (done once at startup) ─────────────────────────────────────
try:
    import torch as _torch
    _cuda_available = _torch.cuda.is_available()
    _cuda_device_name = _torch.cuda.get_device_name(0) if _cuda_available else ""
except Exception:
    _cuda_available = False
    _cuda_device_name = ""

# ── ANSI / terminal-escape stripping ──────────────────────────────────────────
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\x1b[()][0-9A-Za-z]|\r')

def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and carriage-return characters."""
    return _ANSI_RE.sub('', text)

mimetypes.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Temp-file and device helpers
# ─────────────────────────────────────────────────────────────────────────────
_TEMP_DIR = _APP_DIR / "temp"

def _get_temp_dir() -> Path | None:
    """Return the disk temp directory (creating it if needed).

    Returns None when the user has chosen to keep temporary files in RAM
    (i.e. *_use_ram_temp_var* is True), signalling callers to use in-memory
    alternatives instead.
    """
    if _use_ram_temp_var is not None:
        try:
            if _use_ram_temp_var.get():
                return None  # RAM mode – no disk directory needed
        except Exception:
            pass
    _TEMP_DIR.mkdir(exist_ok=True)
    return _TEMP_DIR


def _get_device() -> str:
    """Return 'cuda' or 'cpu' based on the user's GPU/CPU toggle.

    Falls back to 'cpu' when CUDA is not available regardless of the toggle.
    """
    if _cuda_available and _gpu_device_var is not None:
        try:
            if _gpu_device_var.get():
                return 'cuda'
        except Exception:
            pass
    return 'cpu'


# ─────────────────────────────────────────────────────────────────────────────
#  PCM audio helpers  (Live Video tab – sounddevice / numpy streaming)
# ─────────────────────────────────────────────────────────────────────────────

# Named constants
_AUDIO_EXTRACTION_TIMEOUT_SECS = 120   # max seconds to wait for ffmpeg extraction
_PCM_SAMPLE_RATE       = 44100         # output sample rate (Hz)
_PCM_CHANNELS          = 2             # stereo output
_PCM_BLOCK_SIZE        = 2048          # frames per sounddevice callback
_PCM_BYTES_PER_SAMPLE  = 2             # int16 → 2 bytes per sample

# PCM streaming state (shared between video thread and audio callback thread)
_pcm_data         = None   # np.ndarray shape (N, _PCM_CHANNELS) float32
_pcm_pos          = [0]    # current read position in _pcm_data (samples)
_pcm_seek_to      = [-1]   # pending seek target (samples); -1 = none
_pcm_stream       = None   # sounddevice.OutputStream instance
_pcm_speed        = [1.0]  # current playback speed ratio (updated per video frame)
_pcm_volume       = [1.0]  # current volume (0.0 – 1.0+)
_pcm_paused       = [False]  # True while video is paused


def _pcm_extract(video_path: str):
    """Extract raw PCM audio from *video_path* via ffmpeg.

    Audio is decoded to interleaved signed-16-bit little-endian PCM at
    *_PCM_SAMPLE_RATE* Hz with *_PCM_CHANNELS* channels, then normalised to a
    float32 numpy array in the range [-1, 1].

    Returns
    -------
    np.ndarray of shape (N, _PCM_CHANNELS) and dtype float32, or None on
    failure (no audio track, ffmpeg missing, etc.).
    """
    try:
        import numpy as np
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(_PCM_SAMPLE_RATE),
                "-ac", str(_PCM_CHANNELS),
                "-f", "s16le", "pipe:1",
            ],
            capture_output=True,
            timeout=_AUDIO_EXTRACTION_TIMEOUT_SECS,
        )
        if result.returncode == 0 and len(result.stdout) >= _PCM_CHANNELS * _PCM_BYTES_PER_SAMPLE:
            raw = np.frombuffer(result.stdout, dtype=np.int16)
            # Discard any trailing incomplete frame
            n_complete = (len(raw) // _PCM_CHANNELS) * _PCM_CHANNELS
            raw = raw[:n_complete]
            data = raw.reshape(-1, _PCM_CHANNELS).astype(np.float32) / 32768.0
            return data
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return None


def _pcm_audio_callback(outdata, frames, time_info, status):
    """sounddevice OutputStream callback.

    Reads *frames* × speed source samples from *_pcm_data*, resamples them to
    exactly *frames* output samples using linear interpolation, and writes the
    result to *outdata*.  Speed changes take effect immediately each callback
    without any re-extraction or re-encoding step.
    """
    import numpy as np

    # Apply any pending seek requested from the video thread
    seek = _pcm_seek_to[0]
    if seek >= 0:
        _pcm_pos[0] = seek
        _pcm_seek_to[0] = -1

    if _pcm_paused[0] or _pcm_data is None:
        outdata[:] = 0
        return

    speed  = max(0.05, float(_pcm_speed[0]))
    volume = max(0.0,  min(2.0, float(_pcm_volume[0])))
    pos    = _pcm_pos[0]
    total  = len(_pcm_data)

    if pos >= total:
        outdata[:] = 0
        return

    # How many source samples correspond to *frames* output samples at this speed
    n_consume = max(1, round(frames * speed))
    end       = min(pos + n_consume, total)
    n_actual  = end - pos

    chunk = _pcm_data[pos:end]  # shape (n_actual, _PCM_CHANNELS)

    if n_actual == frames:
        # Speed is essentially 1.0 – no interpolation needed
        out = chunk.copy()
    else:
        # Resample n_actual source samples → frames output samples via linear
        # interpolation independently on each channel.
        x_src = np.arange(n_actual, dtype=np.float32)
        x_out = np.linspace(0.0, float(n_actual - 1), frames, dtype=np.float32)
        n_ch  = chunk.shape[1]
        out   = np.empty((frames, n_ch), dtype=np.float32)
        for ch in range(n_ch):
            out[:, ch] = np.interp(x_out, x_src, chunk[:, ch])

    out *= volume
    np.clip(out, -1.0, 1.0, out=out)

    # sounddevice may give us a different channel count than the source
    if outdata.shape[1] <= out.shape[1]:
        outdata[:] = out[:, :outdata.shape[1]]
    else:
        outdata[:, :out.shape[1]] = out
        outdata[:, out.shape[1]:] = 0

    _pcm_pos[0] = end


def _pcm_start_stream(pcm_data, start_pos_secs: float = 0.0, volume: float = 1.0) -> bool:
    """Begin PCM streaming from *start_pos_secs* in the source audio.

    Stores *pcm_data* in the module-level globals, creates and starts a
    sounddevice OutputStream, and returns True on success.
    """
    global _pcm_data, _pcm_stream
    try:
        import sounddevice as sd
        import numpy as np

        _pcm_stop_stream()

        _pcm_data       = pcm_data
        _pcm_pos[0]     = max(0, int(start_pos_secs * _PCM_SAMPLE_RATE))
        _pcm_seek_to[0] = -1
        _pcm_paused[0]  = False
        _pcm_volume[0]  = volume

        _pcm_stream = sd.OutputStream(
            samplerate=_PCM_SAMPLE_RATE,
            channels=_PCM_CHANNELS,
            dtype="float32",
            blocksize=_PCM_BLOCK_SIZE,
            callback=_pcm_audio_callback,
        )
        _pcm_stream.start()
        return True
    except Exception:
        return False


def _pcm_stop_stream() -> None:
    """Stop and close the active sounddevice stream (if any)."""
    global _pcm_stream
    if _pcm_stream is not None:
        try:
            _pcm_stream.stop()
            _pcm_stream.close()
        except Exception:
            pass
        _pcm_stream = None


def _pcm_seek(pos_secs: float) -> None:
    """Schedule a seek to *pos_secs* in the original audio timeline.

    The seek is applied on the next callback invocation so it is safe to call
    from any thread without holding a lock.
    """
    _pcm_seek_to[0] = max(0, int(pos_secs * _PCM_SAMPLE_RATE))


def _pcm_set_volume(volume: float) -> None:
    """Update the playback volume (0.0 – 1.0).  Thread-safe."""
    _pcm_volume[0] = max(0.0, min(1.0, float(volume)))


def _cleanup_live_audio() -> None:
    """Stop PCM streaming and release audio resources."""
    global _pcm_data
    _pcm_stop_stream()
    _pcm_data        = None
    _pcm_pos[0]      = 0
    _pcm_seek_to[0]  = -1
    _pcm_speed[0]    = 1.0
    _pcm_volume[0]   = 1.0
    _pcm_paused[0]   = False


# ─────────────────────────────────────────────────────────────────────────────
#  Training YAML auto-detection helper
# ─────────────────────────────────────────────────────────────────────────────
def _auto_load_training_yaml(folder_path: str) -> None:
    """Search *folder_path* for data.yaml and, if found, auto-fill the Train tab.

    Sets roboflow_yaml_path, patches the YAML to use absolute paths (writing
    the result to *data_training.yaml* so the original *data.yaml* is never
    modified), and populates the class-names textbox (if visible).
    """
    global roboflow_yaml_path
    try:
        from src.dataset import _find_yaml, parse_names, _patch_yaml
        import yaml as _yaml

        folder = Path(folder_path)
        yaml_path = _find_yaml(folder)
        if yaml_path is None:
            return

        with open(str(yaml_path), "r", encoding="utf-8") as _f:
            data = _yaml.safe_load(_f) or {}

        names = parse_names(data.get("names", []))
        if not names:
            return

        # Patch YAML to use absolute paths, writing to a *copy* so the
        # original data.yaml is never modified.
        patched_path = folder / "data_training.yaml"
        _patch_yaml(yaml_path, folder, output_path=patched_path)
        roboflow_yaml_path = str(patched_path)

        # Update class-names textbox if the Train tab is currently open
        if _train_class_names_text is not None:
            try:
                if _train_class_names_text.winfo_exists():
                    _train_class_names_text.delete("1.0", "end")
                    _train_class_names_text.insert("1.0", "\n".join(names))
            except Exception:
                pass

        # Update the Roboflow status label if visible
        if _train_rf_status_label is not None:
            try:
                preview = ", ".join(names[:6]) + ("…" if len(names) > 6 else "")
                _safe_label_configure(
                    _train_rf_status_label,
                    text=f"✅  data.yaml found  •  {len(names)} classes: {preview}",
                    text_color="#4caf50",
                )
            except Exception:
                pass

        messagebox.showinfo(
            "data.yaml Found",
            f"A data.yaml file was found in the selected folder.\n\n"
            f"Classes ({len(names)}): "
            f"{', '.join(names[:8])}{'…' if len(names) > 8 else ''}\n\n"
            "Class names have been filled in automatically.\n"
            "A patched copy (data_training.yaml) has been created for training.\n"
            "The original data.yaml has not been modified.",
        )
    except Exception:
        pass  # silently skip – manual entry still works


# ─────────────────────────────────────────────────────────────────────────────
#  Tooltip helper
# ─────────────────────────────────────────────────────────────────────────────
class Tooltip:
    """Show a brief help tip when the user hovers over any tk/ctk widget."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        try:
            x = self.widget.winfo_rootx() + 24
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffc0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 10),
            wraplength=360,
            padx=7,
            pady=5,
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_screen_size():
    """Return (width, height) of the primary monitor.  Cross-platform."""
    try:
        if sys.platform == "win32":
            import ctypes as _c
            u = _c.windll.user32
            return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        import tkinter as _tk
        _r = _tk.Tk()
        _r.withdraw()
        w, h = _r.winfo_screenwidth(), _r.winfo_screenheight()
        _r.destroy()
        return w, h
    except Exception:
        return 1280, 800


def normalize_path(path: str) -> str:
    if not path:
        return path
    return str(Path(path).resolve())


def clear_frame(frame) -> None:
    for widget in frame.winfo_children():
        widget.destroy()


def _safe_label_configure(label, **kwargs) -> None:
    """Update a label widget only if it still exists."""
    try:
        if label is not None and label.winfo_exists():
            label.configure(**kwargs)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Model catalogue
# ─────────────────────────────────────────────────────────────────────────────
DETECTION_MODELS = [
    "YOLOv8-Nano",      "YOLOv8-Small",     "YOLOv8-Medium",
    "YOLOv8-Large",     "YOLOv8-ExtraLarge",
    "YOLOv9-Compact",   "YOLOv9-Enhanced",
    "YOLOv10-Nano",     "YOLOv10-Small",    "YOLOv10-Medium",
    "YOLOv10-Balanced", "YOLOv10-Large",    "YOLOv10-ExtraLarge",
    "YOLOv11-Nano",     "YOLOv11-Small",    "YOLOv11-Medium",
    "YOLOv11-Large",    "YOLOv11-ExtraLarge",
    "YOLOv12-Nano",     "YOLOv12-Small",    "YOLOv12-Medium",
    "YOLOv12-Large",    "YOLOv12-ExtraLarge",
]

SEGMENTATION_MODELS = [
    "YOLOv8-Nano-Seg",       "YOLOv8-Small-Seg",      "YOLOv8-Medium-Seg",
    "YOLOv8-Large-Seg",      "YOLOv8-ExtraLarge-Seg",
    "YOLOv11-Nano-Seg",      "YOLOv11-Small-Seg",     "YOLOv11-Medium-Seg",
    "YOLOv11-Large-Seg",     "YOLOv11-ExtraLarge-Seg",
]

CLASSIFICATION_MODELS = [
    "YOLOv8-Nano-Cls",       "YOLOv8-Small-Cls",      "YOLOv8-Medium-Cls",
    "YOLOv8-Large-Cls",      "YOLOv8-ExtraLarge-Cls",
    "YOLOv11-Nano-Cls",      "YOLOv11-Small-Cls",     "YOLOv11-Medium-Cls",
    "YOLOv11-Large-Cls",     "YOLOv11-ExtraLarge-Cls",
]

POSE_MODELS = [
    "YOLOv8-Nano-Pose",      "YOLOv8-Small-Pose",     "YOLOv8-Medium-Pose",
    "YOLOv8-Large-Pose",     "YOLOv8-ExtraLarge-Pose",
    "YOLOv11-Nano-Pose",     "YOLOv11-Small-Pose",    "YOLOv11-Medium-Pose",
    "YOLOv11-Large-Pose",    "YOLOv11-ExtraLarge-Pose",
]

OBB_MODELS = [
    "YOLOv8-Nano-OBB",       "YOLOv8-Small-OBB",      "YOLOv8-Medium-OBB",
    "YOLOv8-Large-OBB",      "YOLOv8-ExtraLarge-OBB",
    "YOLOv11-Nano-OBB",      "YOLOv11-Small-OBB",     "YOLOv11-Medium-OBB",
    "YOLOv11-Large-OBB",     "YOLOv11-ExtraLarge-OBB",
]

TASK_TYPE_OPTIONS = ["Detection", "Segmentation", "Classification", "Pose Estimation", "OBB Detection"]

_MODELS_BY_TASK = {
    "Detection":       DETECTION_MODELS,
    "Segmentation":    SEGMENTATION_MODELS,
    "Classification":  CLASSIFICATION_MODELS,
    "Pose Estimation": POSE_MODELS,
    "OBB Detection":   OBB_MODELS,
}

MODEL_MAP = {
    # Detection
    "YOLOv8-Nano":       "yolov8n",   "YOLOv8-Small":       "yolov8s",
    "YOLOv8-Medium":     "yolov8m",   "YOLOv8-Large":       "yolov8l",
    "YOLOv8-ExtraLarge": "yolov8x",
    "YOLOv9-Compact":    "yolov9c",   "YOLOv9-Enhanced":    "yolov9e",
    "YOLOv10-Nano":      "yolov10n",  "YOLOv10-Small":      "yolov10s",
    "YOLOv10-Medium":    "yolov10m",  "YOLOv10-Balanced":   "yolov10b",
    "YOLOv10-Large":     "yolov10l",  "YOLOv10-ExtraLarge": "yolov10x",
    "YOLOv11-Nano":      "yolo11n",   "YOLOv11-Small":      "yolo11s",
    "YOLOv11-Medium":    "yolo11m",   "YOLOv11-Large":      "yolo11l",
    "YOLOv11-ExtraLarge":"yolo11x",
    "YOLOv12-Nano":      "yolo12n",   "YOLOv12-Small":      "yolo12s",
    "YOLOv12-Medium":    "yolo12m",   "YOLOv12-Large":      "yolo12l",
    "YOLOv12-ExtraLarge":"yolo12x",
    # Segmentation
    "YOLOv8-Nano-Seg":        "yolov8n-seg",  "YOLOv8-Small-Seg":      "yolov8s-seg",
    "YOLOv8-Medium-Seg":      "yolov8m-seg",  "YOLOv8-Large-Seg":      "yolov8l-seg",
    "YOLOv8-ExtraLarge-Seg":  "yolov8x-seg",
    "YOLOv11-Nano-Seg":       "yolo11n-seg",  "YOLOv11-Small-Seg":     "yolo11s-seg",
    "YOLOv11-Medium-Seg":     "yolo11m-seg",  "YOLOv11-Large-Seg":     "yolo11l-seg",
    "YOLOv11-ExtraLarge-Seg": "yolo11x-seg",
    # Classification
    "YOLOv8-Nano-Cls":        "yolov8n-cls",  "YOLOv8-Small-Cls":      "yolov8s-cls",
    "YOLOv8-Medium-Cls":      "yolov8m-cls",  "YOLOv8-Large-Cls":      "yolov8l-cls",
    "YOLOv8-ExtraLarge-Cls":  "yolov8x-cls",
    "YOLOv11-Nano-Cls":       "yolo11n-cls",  "YOLOv11-Small-Cls":     "yolo11s-cls",
    "YOLOv11-Medium-Cls":     "yolo11m-cls",  "YOLOv11-Large-Cls":     "yolo11l-cls",
    "YOLOv11-ExtraLarge-Cls": "yolo11x-cls",
    # Pose Estimation
    "YOLOv8-Nano-Pose":       "yolov8n-pose", "YOLOv8-Small-Pose":     "yolov8s-pose",
    "YOLOv8-Medium-Pose":     "yolov8m-pose", "YOLOv8-Large-Pose":     "yolov8l-pose",
    "YOLOv8-ExtraLarge-Pose": "yolov8x-pose",
    "YOLOv11-Nano-Pose":      "yolo11n-pose", "YOLOv11-Small-Pose":    "yolo11s-pose",
    "YOLOv11-Medium-Pose":    "yolo11m-pose", "YOLOv11-Large-Pose":    "yolo11l-pose",
    "YOLOv11-ExtraLarge-Pose": "yolo11x-pose",
    # OBB Detection
    "YOLOv8-Nano-OBB":        "yolov8n-obb",  "YOLOv8-Small-OBB":      "yolov8s-obb",
    "YOLOv8-Medium-OBB":      "yolov8m-obb",  "YOLOv8-Large-OBB":      "yolov8l-obb",
    "YOLOv8-ExtraLarge-OBB":  "yolov8x-obb",
    "YOLOv11-Nano-OBB":       "yolo11n-obb",  "YOLOv11-Small-OBB":     "yolo11s-obb",
    "YOLOv11-Medium-OBB":     "yolo11m-obb",  "YOLOv11-Large-OBB":     "yolo11l-obb",
    "YOLOv11-ExtraLarge-OBB": "yolo11x-obb",
}

EXPORT_FORMATS = ["ONNX", "TensorRT Engine", "CoreML", "TF SavedModel", "TFLite"]
EXPORT_FORMAT_MAP = {
    "ONNX":            "onnx",
    "TensorRT Engine": "engine",
    "CoreML":          "coreml",
    "TF SavedModel":   "saved_model",
    "TFLite":          "tflite",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Global application state
# ─────────────────────────────────────────────────────────────────────────────
project_name                 = ""
train_data_path              = ""
model_save_path              = ""
custom_model_path            = ""    # optional .pt for training base
roboflow_yaml_path           = ""    # set when a Roboflow ZIP is imported
input_size                   = ""
epochs                       = ""
batch_size                   = ""
class_names                  = []
image_paths                  = []
current_image_index          = 0
detection_model_path         = ""
detection_images_folder_path = ""
detection_save_dir           = ""
export_model_path            = ""
camera_detection             = None

# Widget references populated inside show_* functions
output_textbox         = None
progress_bar           = None
detection_progress_bar = None
image_label            = None
image_index_label      = None
selected_model_var     = None   # StringVar for model dropdown
task_type_var          = None   # StringVar "Detection" / "Segmentation"
model_menu_widget      = None   # CTkOptionMenu reference
_camera_bar            = None   # bottom bar in camera view
camera_id_entry        = None

# Status labels (set inside each show_* function, None when not visible)
train_data_label    = None
model_save_label    = None
custom_model_label  = None
detect_folder_label = None
detect_model_label  = None
export_model_label  = None
export_status_label = None

# Detect tab extended state
_detection_running        = False
_detection_cancel_flag    = [False]
_detect_start_btn         = None        # reference to start/cancel button
_detect_controls          = []          # widgets to disable during detection
_detect_file_count_label  = None
_detect_model_info_label  = None
_detect_conf_var          = None        # DoubleVar for confidence threshold
_detect_half_var          = None        # BooleanVar for FP16
_detect_workers_var       = None        # StringVar for workers
_detect_task_var          = None        # StringVar for model task override
_detect_progress_label    = None        # text progress label
_detect_nav_bar           = None        # bottom nav bar in detect view
_detect_zoom_var          = None        # DoubleVar – image preview zoom level

# Train tab widget references
_train_class_names_text   = None    # CTkTextbox – class names in Train tab
_train_rf_status_label    = None    # status label for YAML auto-detection

# Train tab extra-params widget references
_train_time_var          = None   # DoubleVar – time (hours)
_train_patience_var      = None   # StringVar – patience
_train_save_var          = None   # BooleanVar – save
_train_save_period_var   = None   # StringVar – save_period
_train_cache_var         = None   # BooleanVar – cache
_train_resume_var        = None   # BooleanVar – resume
_train_freeze_var        = None   # StringVar – freeze
_train_lr0_var           = None   # StringVar – lr0
_train_lrf_var           = None   # StringVar – lrf
_train_momentum_var      = None   # StringVar – momentum
_train_weight_decay_var  = None   # StringVar – weight_decay
_train_optimizer_var     = None   # StringVar – optimizer
_train_val_var           = None   # BooleanVar – val
_train_max_det_var       = None   # StringVar – max_det
_train_max_det_widget    = None   # widget reference for enabling/disabling

# Training process control
_train_proc              = [None]    # [subprocess.Popen] current training process
_train_stop_btn_ref      = [None]    # [CTkButton] stop training button

# Browse-button last-dir memory and button references
_browse_last_dirs        = {}     # key → last visited directory string
_train_data_btn_ref      = [None]
_model_save_btn_ref      = [None]
_custom_model_btn_ref    = [None]

# Camera tab state
_camera_half_var          = None    # BooleanVar – FP16 for camera inference

# Live-video tab state
_live_video_path          = ""
_live_video_running       = False
_live_video_cancel_flag   = [False]
_live_video_label         = None
_live_video_status_label  = None
_live_video_start_btn     = None
_live_video_bar           = None
_live_video_paused        = False
_live_video_pause_btn     = None
_live_video_seek_slider   = None
_live_video_seek_to       = [-1]    # frame to seek to; -1 = no pending seek
_live_video_frame_ref     = [0]     # current frame index (updated by thread)
_live_video_total_ref     = [1]     # total frames (set by thread at startup)
_live_video_fps_ref       = [25.0]  # video fps (set by thread at startup)
_live_video_seeking       = [False] # True while slider drag is in flight
_live_video_raw_frame     = [None]  # latest raw BGR frame (screenshots)
_live_video_ann_frame     = [None]  # latest annotated BGR frame (screenshots)
_live_video_half_var      = None    # BooleanVar – FP16 inference
_live_video_conf_var      = None    # DoubleVar  – confidence threshold
_live_video_task_var      = None    # StringVar  – model task override (for TensorRT/ONNX)

# Live-video audio state
_live_audio_enabled_var   = None    # BooleanVar – enable audio playback
_live_audio_sync_var      = None    # BooleanVar – sync audio speed to video FPS
_live_audio_volume_var    = None    # DoubleVar  – playback volume (0.0–1.0)
_live_video_is_url        = [False] # True when source is a URL/stream

# Sidebar settings state (set once when sidebar is built)
_use_ram_temp_var         = None    # BooleanVar – store temp files in RAM
_gpu_device_var           = None    # BooleanVar – True = use GPU, False = CPU

# Training queue state
_train_queue              = []          # list of dict with training job configs
_train_queue_frame        = None        # frame listing queued jobs
_train_queue_running      = False

# Benchmark state
_benchmark_models           = []   # list of .pt paths added by user
_benchmark_results_frame    = None
_benchmark_run_btn          = None
_benchmark_model_list_frame = None

output_queue = Queue()

# ─────────────────────────────────────────────────────────────────────────────
#  Persistent tab-state buffers  (survive tab switches)
# ─────────────────────────────────────────────────────────────────────────────
_train_log_buffer         = []     # accumulated training output lines
_train_progress_value     = 0.0   # last progress bar fraction for train tab
_train_progress_text      = ""    # last progress label text for train tab

_detect_file_count_text   = ""    # last file-count label text
_detect_model_info_text   = ""    # last model-info label text
_detect_progress_value    = 0.0   # last detection progress bar fraction
_detect_progress_text     = ""    # last detection progress label text


# ─────────────────────────────────────────────────────────────────────────────
#  Embedded logo  (base64-encoded PNG, displayed in sidebar)
# ─────────────────────────────────────────────────────────────────────────────
_LOGO_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAHgAAAB4CAYAAAA5ZDbSAAAHA0lEQVR42u3dr3bbOhgAcMkn3AMZ"
    "LbmkuCMDDR0pCth9gj5EydDIHqJPsIKgkNEUjCR45JJLFzA/wQZSZ7Zjy/rz/VX1ndOz0y6Jbf3y"
    "SZZsS8aUKFGiRIkSJTjCZnlU33//jn7ve2sLcC6YrwDdFtC8wW2OqPV61cRuqtns6pywrXbYFExQ"
    "dKHQVhtqLGiz7yPV7xCzXBD2QgMsRZbGNgGj2O2xCIBeSMaVBuvax0loZuRFgUWGZs7mRYHNG3oh"
    "ATcH2CBoQuQFJ26OsF7QhMgVGewrxHUe70iZ6AR+JVVyFLJHv1/2QAdj1jb7h2P/L3f/9H/f/ncB"
    "8O7Lkmz/hm0zUpVtc8C9xByLeeDRrENEp0C2WnHnUGNgMD6TG9lqw51CQCl8om1hIlstuMPCpmwv"
    "KfYBC9lKx5UAS7VPGMhWKq5E2Ml9/P/u+HLc19KQrXRcibADkB/GGGOutkuo/YVEBgVOxdUEOwSu"
    "16tryP3vIScAVwUXBne43359c0eXrFueCaNdFQTua6qSQwZGUpEhyttCbCzpLsaXQtAGO8xerGNL"
    "bY+TgSXjftzsnn1f+7Re3UIDoyCjAwvGDQFNAffFlYDMcssONO4UbEhWDj+j/T00s6fa5Wb/cGz2"
    "D0fqpshSZy8U7hgqBMbc54dmL+Sxx2SxjcGNBcbChYSd29ajMctY4NQyiDnhiqqiU3C1wA638XGz"
    "e25x7405PgE0U6HI9XrVhD47VcVkL1Q/MRb3ab26pcAdQtfr1fW9MceUEzrQNtjDpaLMXihczv5v"
    "d/upyDG1Wmj529BvSugGoHC5YTH2LbZsQk62KkxcY8z5UlpuuMP2mXK7IePUqLfNdq+Xni+rZYIL"
    "gZxSVcOcZEGMWl1tl22XotnsfvhAa8HlzGTfLK6ws7f9ltbr1XUXOhfcVGTsLK6oC6KFHstm6raM"
    "anCEMyquM2dXta0te1P3OzaLfarpirNAhtX2ozFLrbjcZ9ZkVXRM3+7emGM7QpRThCBjtcWVpAKB"
    "uO1Uc1WNEQu07pGg+PXpeXKA/s3nW9XH17sAMfJgOWgGx1QvmN2iX5+eaxeu72uo22LIahrljg7u"
    "G+guwG4cLz7038OZ0e2dH9m0wRjZ28O9mcEdeQ1kNks4oxZ1kgWOGxJIyPK6SUpPsJJwlSO7BjzA"
    "MljMDew3zO8HOn+BaovZqmjI9vecbQ6ct/9+uPiZQ4bIYu52OKs22IUb8vecQj3wXPbOIU7+P2AW"
    "c4ZXP3g4mfZU83H650sjPXPHXvfz67csM3hhSgiM05xezf7O46TM3dOpSmGWNrhE7lW0zwIWnX7b"
    "shRraYNJ4ufXb14nWvJOsE7zaEIMGqmvos9Xfw5xeJP/fxh8fmmDZWeyjszNCBhyCG8ui1vM4c9k"
    "AGYv933eYMAUj2F4xYH5/YkBfdHmErhzT0/wQo0S2uIUpMPE5wkP19OGWbXBSchKcUWfZGFcSrtA"
    "PnjAIuFKeM4KpR/MMV3QGPL5StAh8IvB3P6KzeAYVMwL4m8+3zZzcD6voc5eyOQYz+D31rb39jSb"
    "Xa395nfq7Ax92B3rBEvcSVYuj48ac5pP61HAuDw4cEx/WPsThcPoPkznO6sB1k2LYjJY2mOXEG2v"
    "76wGPBmcMOCROqqlFXlqv12zGqRkr890SqLa4Fyq6qnjCJ2MBiLcE6ElPuUQ+83UOBFL6D73gK+2"
    "y+TJSaMymGlcWlt7HPOF7LbPGN0jkio6pS3WgpxS2zT7h2Ns9sKdRTNeXZKOzNWUhMxViT4ZKUQf"
    "T9Jss1D7AzYxePJkpK4PJ+o2QUzhmyUu2EAH4Hq2EMiU3YwuLCQuSHi4qFuzoYXtzqtFuWZD6rao"
    "12yoIL8t2FW1b3ZhfSYnbqwHy8JYCUNzo0vaYKybBF07cCypwwYce8BzaxZRrXzG1DQRATMhxyxI"
    "hbl2oQZcEGAq5JQVx7hC7+qjSMhThaENF2o9ZIil3uPHogH7xmiLKivGhSrvCmqjqePUU8iashcS"
    "N6XdhamiEarq0Sx+WXdJMjB01kJUzTAZPLJxiCtO50JqF9W62i6lwkrGhclgxEwea5PZp0pE3Cdo"
    "XFhgRGQp0Jj7gIELD4yM7DrLxgCn2hYWLg4wAbJPdyrqag3CZ3Li4gETIYf3nU8zyP2N02w21KhU"
    "uLjAxMh+6G5gynadAhcfeASZGrqP3i9Unwne0GERcWH6wYH9ZKi+ssagxqUBbg8CYUBENe5ImWAE"
    "7VSGnQfLuwedw+pqkrKWD3gEOVfoyRqKEJcHuHuQGUJLgeUFzhBaGqwMYEe1rQXaebLIjCsH2JHN"
    "w0KUgD3bAxAASzfQATxIMgwKcK8unSBUPcAB0FDowf1zobC6gBOwUZsTBaEPmANcEWh+wNDoijFL"
    "lChRooSq+AMg9whXo2gm/wAAAABJRU5ErkJggg=="
)


# ─────────────────────────────────────────────────────────────────────────────
#  Output-queue consumer  (runs on main thread via root.after)
# ─────────────────────────────────────────────────────────────────────────────
def update_output_textbox() -> None:
    global output_textbox, _train_log_buffer
    try:
        line = output_queue.get_nowait()
        _train_log_buffer.append(line)
        if output_textbox is not None and output_textbox.winfo_exists():
            output_textbox.insert("end", line)
            output_textbox.yview_moveto(1)
    except Empty:
        pass
    except Exception:
        pass
    finally:
        root.after(100, update_output_textbox)


# ─────────────────────────────────────────────────────────────────────────────
#  Image navigation (detection results)
# ─────────────────────────────────────────────────────────────────────────────
def update_image() -> None:
    global current_image_index, image_label, image_paths, image_index_label
    if not image_paths or image_label is None:
        return
    try:
        if not image_label.winfo_exists():
            return
    except Exception:
        return

    _safe_label_configure(image_index_label, text=f"{current_image_index + 1}/{len(image_paths)}")
    img = Image.open(image_paths[current_image_index])
    img_w, img_h = img.size
    max_w = max(1, image_label.winfo_width())
    max_h = max(1, image_label.winfo_height())
    zoom = _detect_zoom_var.get() if _detect_zoom_var is not None else 1.0
    scale = min(max_w / img_w, max_h / img_h) * zoom
    img = img.resize(
        (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
        Image.Resampling.LANCZOS,
    )
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo  # keep reference alive


def show_next_image() -> None:
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index + 1) % len(image_paths)
        update_image()


def show_prev_image() -> None:
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index - 1) % len(image_paths)
        update_image()


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def on_sidebar_select(key: str) -> None:
    global train_data_label, model_save_label, custom_model_label
    global detect_folder_label, detect_model_label
    global export_model_label, export_status_label
    global output_textbox, progress_bar, detection_progress_bar
    global image_label, image_index_label, model_menu_widget
    global selected_model_var, task_type_var, _camera_bar
    global _benchmark_results_frame, _benchmark_run_btn, _benchmark_model_list_frame
    global _detect_start_btn, _detect_controls, _detect_file_count_label
    global _detect_model_info_label, _detect_conf_var, _detect_half_var
    global _detect_workers_var, _detect_task_var, _detect_progress_label, _detect_nav_bar
    global _detect_zoom_var
    global _live_video_label, _live_video_status_label, _live_video_start_btn, _live_video_bar
    global _live_video_pause_btn, _live_video_seek_slider, _live_video_half_var
    global _live_video_conf_var, _live_video_paused
    global _live_audio_enabled_var, _live_audio_sync_var, _live_audio_volume_var
    global _train_queue_frame
    global _train_class_names_text, _train_rf_status_label
    global _camera_half_var
    global _train_time_var, _train_patience_var, _train_save_var
    global _train_save_period_var, _train_cache_var, _train_resume_var
    global _train_freeze_var, _train_lr0_var, _train_lrf_var
    global _train_momentum_var, _train_weight_decay_var, _train_optimizer_var
    global _train_val_var, _train_max_det_var, _train_max_det_widget

    # Stop live video playback and clean up audio if running
    _live_video_cancel_flag[0] = True
    _cleanup_live_audio()

    clear_frame(main_frame)

    train_data_label = model_save_label = custom_model_label = None
    detect_folder_label = detect_model_label = None
    export_model_label = export_status_label = None
    image_label = image_index_label = None
    model_menu_widget = None
    _camera_bar = None
    _benchmark_results_frame = None
    _benchmark_run_btn = None
    _benchmark_model_list_frame = None
    _detect_start_btn = None
    _detect_controls = []
    _detect_file_count_label = None
    _detect_model_info_label = None
    _detect_conf_var = None
    _detect_half_var = None
    _detect_workers_var = None
    _detect_task_var = None
    _detect_progress_label = None
    _detect_nav_bar = None
    _detect_zoom_var = None
    _live_video_label = None
    _live_video_status_label = None
    _live_video_start_btn = None
    _live_video_bar = None
    _live_video_pause_btn = None
    _live_video_seek_slider = None
    _live_video_half_var = None
    _live_video_conf_var = None
    _live_video_paused = False
    _live_audio_enabled_var = None
    _live_audio_sync_var = None
    _live_audio_volume_var = None
    _train_queue_frame = None
    _train_class_names_text = None
    _train_rf_status_label = None
    _camera_half_var = None
    _train_time_var = _train_patience_var = _train_save_var = None
    _train_save_period_var = _train_cache_var = _train_resume_var = None
    _train_freeze_var = _train_lr0_var = _train_lrf_var = None
    _train_momentum_var = _train_weight_decay_var = _train_optimizer_var = None
    _train_val_var = _train_max_det_var = _train_max_det_widget = None
    _train_stop_btn_ref[0] = None

    if key == "Train":
        show_ai_train_window()
    elif key == "Detect":
        show_image_detection_window()
    elif key == "Camera":
        show_camera_detection_window()
    elif key == "LiveVideo":
        show_live_video_window()
    elif key == "Export":
        show_export_window()
    elif key == "Benchmark":
        show_benchmark_window()


# ─────────────────────────────────────────────────────────────────────────────
#  Train window
# ─────────────────────────────────────────────────────────────────────────────
def _on_task_type_change(*_args) -> None:
    global model_menu_widget, selected_model_var, task_type_var
    if task_type_var is None or model_menu_widget is None:
        return
    task = task_type_var.get()
    options = _MODELS_BY_TASK.get(task, DETECTION_MODELS)
    selected_model_var.set(options[0])
    model_menu_widget.configure(values=options)


def show_ai_train_window() -> None:
    global output_textbox, progress_bar, selected_model_var, task_type_var, model_menu_widget
    global train_data_label, model_save_label, custom_model_label
    global _train_time_var, _train_patience_var, _train_save_var
    global _train_save_period_var, _train_cache_var, _train_resume_var
    global _train_freeze_var, _train_lr0_var, _train_lrf_var
    global _train_momentum_var, _train_weight_decay_var, _train_optimizer_var
    global _train_val_var, _train_max_det_var, _train_max_det_widget
    global _train_data_btn_ref, _model_save_btn_ref, _custom_model_btn_ref
    global _train_stop_btn_ref

    # ── Left: scrollable configuration panel ──────────────────────────────
    config_panel = ctk.CTkScrollableFrame(
        master=main_frame,
        label_text="Training Configuration",
        label_font=("Segoe UI", 14, "bold"),
        corner_radius=8,
    )
    config_panel.place(relx=0, rely=0, relwidth=0.41, relheight=1.0)

    # ── Right: log / output panel ──────────────────────────────────────────
    log_panel = ctk.CTkFrame(master=main_frame, corner_radius=8)
    log_panel.place(relx=0.42, rely=0, relwidth=0.58, relheight=1.0)

    PAD  = {"padx": 14, "pady": 5}
    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)
    FENT = ("Segoe UI", 13)

    def _lbl(text: str):
        l = ctk.CTkLabel(config_panel, text=text, font=FLAB, anchor="w")
        l.pack(fill="x", padx=14, pady=(8, 1))
        return l

    def _sep():
        ctk.CTkFrame(config_panel, height=1, fg_color="gray50").pack(
            fill="x", padx=14, pady=4
        )

    # ── Roboflow ZIP import ────────────────────────────────────────────────
    _lbl("📦  Import Roboflow Dataset  (optional)")

    _rf_status_label = ctk.CTkLabel(
        config_panel, text="No Roboflow dataset loaded",
        font=("Segoe UI", 11), text_color="gray", anchor="w",
    )

    global _train_rf_status_label
    _train_rf_status_label = _rf_status_label

    def _do_roboflow_import():
        global train_data_path, roboflow_yaml_path, train_data_label

        zip_path = filedialog.askopenfilename(
            title="Select Roboflow Dataset ZIP",
            filetypes=[("ZIP archive", "*.zip"), ("All files", "*.*")],
        )
        if not zip_path:
            return

        extract_dir = filedialog.askdirectory(
            title="Choose folder to extract dataset into"
        )
        if not extract_dir:
            return

        # Disable the import button while work is in progress
        rf_btn.configure(state="disabled", text="⏳  Importing…")
        _safe_label_configure(_rf_status_label, text="⏳ Starting extraction…", text_color="#64b5f6")
        _rf_progress_bar.set(0)
        _rf_progress_bar.pack(fill="x", padx=14, pady=(2, 0))

        def _progress_cb(current: int, total: int, filename: str) -> None:
            frac = current / max(total, 1)
            msg = f"⏳ Extracting… {current}/{total}  ({frac * 100:.0f}%)"
            root.after(0, lambda f=frac, m=msg: (
                _rf_progress_bar.set(f),
                _safe_label_configure(_rf_status_label, text=m, text_color="#64b5f6"),
            ))

        def _run_import():
            try:
                from src.dataset import extract_roboflow_zip, count_dataset_images
                r_path, yaml, names = extract_roboflow_zip(
                    zip_path, extract_dir, progress_callback=_progress_cb
                )
            except Exception as exc:
                root.after(0, lambda: (
                    messagebox.showerror("Import Error", f"Failed to extract ZIP:\n{exc}"),
                    _safe_label_configure(_rf_status_label, text="Import failed.", text_color="#ef5350"),
                    _rf_progress_bar.pack_forget(),
                    rf_btn.configure(state="normal", text="📦  Import Roboflow ZIP…"),
                ))
                return

            try:
                from src.dataset import count_dataset_images
                counts = count_dataset_images(r_path)
                parts  = [f"{v} {k}" for k, v in counts.items()]
                count_str = "  |  ".join(parts) if parts else "unknown"
            except Exception:
                count_str = "unknown"

            def _apply():
                global train_data_path, roboflow_yaml_path
                train_data_path    = r_path
                roboflow_yaml_path = yaml

                _safe_label_configure(train_data_label, text=Path(r_path).name, text_color="#64b5f6")

                # Auto-fill class names textbox
                class_names_text.delete("1.0", "end")
                class_names_text.insert("1.0", "\n".join(names))

                status = f"✅  {Path(zip_path).stem}  •  {len(names)} classes  •  {count_str} images"
                _safe_label_configure(_rf_status_label, text=status, text_color="#4caf50")
                _rf_progress_bar.set(1.0)
                _rf_progress_bar.pack_forget()
                rf_btn.configure(state="normal", text="📦  Import Roboflow ZIP…")

                preview = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
                messagebox.showinfo(
                    "Dataset Imported Successfully",
                    f"Roboflow dataset extracted to:\n{r_path}\n\n"
                    f"Classes ({len(names)}): {preview}\n\n"
                    f"Images: {count_str}\n\n"
                    "Class names have been filled in automatically.\n"
                    "Select a model, set epochs/batch, and click Start Training.",
                )

            root.after(0, _apply)

        threading.Thread(target=_run_import, daemon=True).start()

    def _clear_roboflow():
        global roboflow_yaml_path
        roboflow_yaml_path = ""
        _safe_label_configure(
            _rf_status_label, text="No Roboflow dataset loaded", text_color="gray"
        )

    rf_btn = ctk.CTkButton(
        config_panel, text="📦  Import Roboflow ZIP…", font=FBTN, height=36,
        command=_do_roboflow_import,
    )
    rf_btn.pack(fill="x", **PAD)
    Tooltip(
        rf_btn,
        "Import a dataset downloaded from Roboflow in YOLOv8 format.\n\n"
        "How to get one: roboflow.com → your project → Export → YOLOv8 → Download ZIP\n\n"
        "Expected ZIP layout:\n"
        "  train/images/*.jpg    train/labels/*.txt\n"
        "  valid/images/*.jpg    valid/labels/*.txt\n"
        "  test/images/*.jpg     test/labels/*.txt  (optional)\n"
        "  data.yaml\n\n"
        "The app will extract, patch paths, and fill in class names automatically.",
    )
    _rf_status_label.pack(fill="x", padx=14)
    # Progress bar for ZIP extraction (hidden until import starts)
    _rf_progress_bar = ctk.CTkProgressBar(
        config_panel, progress_color="#64b5f6", mode="determinate",
    )
    _rf_progress_bar.set(0)
    ctk.CTkButton(
        config_panel, text="Clear imported dataset", font=("Segoe UI", 11),
        height=28, fg_color="gray50", hover_color="gray35",
        command=_clear_roboflow,
    ).pack(fill="x", padx=14, pady=(2, 4))
    _sep()

    # ── Project name ───────────────────────────────────────────────────────
    _lbl("Project Name")
    project_name_entry = ctk.CTkEntry(
        config_panel, placeholder_text="e.g.  my_detector", font=FENT, height=36
    )
    project_name_entry.pack(fill="x", **PAD)
    Tooltip(
        project_name_entry,
        "A short alphanumeric name for this training run.\n"
        "Results and the YAML config are saved under this name.",
    )
    _sep()

    # ── Training data folder ───────────────────────────────────────────────
    _lbl("Training Data Folder")
    train_data_btn = ctk.CTkButton(
        config_panel, text="Browse…", font=FBTN, height=36, command=select_train_data
    )
    train_data_btn.pack(fill="x", **PAD)
    _train_data_btn_ref[0] = train_data_btn
    Tooltip(
        train_data_btn,
        "Select a folder containing your YOLO-format image+annotation pairs.\n\n"
        "Supported layouts (auto-detected):\n"
        "  1. Flat pairs in root:\n"
        "       folder/photo1.jpg  folder/photo1.txt …\n\n"
        "  2. Separate images/ and labels/ sub-folders:\n"
        "       folder/images/photo1.jpg\n"
        "       folder/labels/photo1.txt\n\n"
        "  3. Split sub-folders (any capitalisation):\n"
        "       folder/Train/images/  folder/Train/labels/\n"
        "       folder/Valid/images/  folder/Valid/labels/\n\n"
        "  4. Pre-split train/val:\n"
        "       folder/train/images/  folder/val/images/\n\n"
        "For layouts 1–3 the app automatically creates an 80/20 train/val split.\n\n"
        "If you imported a Roboflow ZIP above, this is set automatically.",
    )
    train_data_label = ctk.CTkLabel(
        config_panel, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    train_data_label.pack(fill="x", padx=14)
    _sep()

    # ── Save folder ────────────────────────────────────────────────────────
    _lbl("Model Save Folder")
    model_save_btn = ctk.CTkButton(
        config_panel, text="Browse…", font=FBTN, height=36, command=select_model_save_folder
    )
    model_save_btn.pack(fill="x", **PAD)
    _model_save_btn_ref[0] = model_save_btn
    Tooltip(model_save_btn, "Choose where the trained model weights and results will be saved.")
    model_save_label = ctk.CTkLabel(
        config_panel, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    model_save_label.pack(fill="x", padx=14)
    _sep()

    # ── Task type ──────────────────────────────────────────────────────────
    _lbl("Task Type")
    task_type_var = ctk.StringVar(value="Detection")
    task_menu_w = ctk.CTkOptionMenu(
        config_panel,
        variable=task_type_var,
        values=TASK_TYPE_OPTIONS,
        command=_on_task_type_change,
        font=FBTN,
        height=36,
    )
    task_menu_w.pack(fill="x", **PAD)
    Tooltip(
        task_menu_w,
        "Detection         – bounding boxes around objects.\n"
        "Segmentation      – pixel-level instance masks (polygon annotations required).\n"
        "Classification    – predict a single class label for the whole image.\n"
        "Pose Estimation   – detect keypoints / skeleton joints.\n"
        "OBB Detection     – oriented (rotated) bounding boxes.",
    )

    # ── YOLO model dropdown ────────────────────────────────────────────────
    _lbl("YOLO Model")
    selected_model_var = ctk.StringVar(value=DETECTION_MODELS[0])
    model_menu_widget = ctk.CTkOptionMenu(
        config_panel,
        variable=selected_model_var,
        values=DETECTION_MODELS,
        font=FBTN,
        dropdown_font=FBTN,
        height=36,
    )
    model_menu_widget.pack(fill="x", **PAD)
    Tooltip(
        model_menu_widget,
        "Pre-trained Ultralytics weights used as the training starting point.\n\n"
        "Nano / Small  – fastest, least accurate; ideal for edge devices.\n"
        "Medium        – balanced speed and accuracy.\n"
        "Large / ExtraLarge – most accurate; needs more GPU memory.\n\n"
        "Segmentation variants require polygon annotations.",
    )
    _sep()

    # ── Custom base model (optional) ───────────────────────────────────────
    _lbl("Custom Base Model  (optional)")
    _custom_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _custom_row.pack(fill="x", **PAD)
    _custom_model_btn = ctk.CTkButton(
        _custom_row, text="Browse .pt…", font=FBTN, height=36, command=select_custom_model
    )
    _custom_model_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
    _custom_model_btn_ref[0] = _custom_model_btn
    Tooltip(
        _custom_model_btn,
        "Load your own .pt file as the training starting point.\n"
        "When set, this overrides the YOLO Model dropdown above.\n\n"
        "Useful for fine-tuning an already-trained custom model.",
    )
    # Resume toggle on same row
    _train_resume_var = ctk.BooleanVar(value=False)
    _resume_switch = ctk.CTkSwitch(
        _custom_row, text="Resume", variable=_train_resume_var,
        font=("Segoe UI", 12), width=80,
    )
    _resume_switch.pack(side="left", padx=(0, 4))
    Tooltip(
        _resume_switch,
        "Resumes training from the last saved checkpoint.\n"
        "Automatically loads model weights, optimizer state, and epoch count,\n"
        "continuing training seamlessly.",
    )
    custom_model_label = ctk.CTkLabel(
        config_panel, text="Using built-in pretrained weights",
        font=("Segoe UI", 11), text_color="gray", anchor="w",
    )
    custom_model_label.pack(fill="x", padx=14)
    ctk.CTkButton(
        config_panel, text="Clear custom model", font=("Segoe UI", 11),
        height=28, fg_color="gray50", hover_color="gray35",
        command=clear_custom_model,
    ).pack(fill="x", padx=14, pady=(2, 4))
    _sep()

    # ── Numeric training params ────────────────────────────────────────────
    _lbl("Image Size  (e.g. 640)")
    input_size_entry = ctk.CTkEntry(
        config_panel, placeholder_text="640", font=FENT, height=36
    )
    input_size_entry.insert(0, "640")
    input_size_entry.pack(fill="x", **PAD)
    Tooltip(
        input_size_entry,
        "Square resolution fed into the network (pixels).\n"
        "640 is standard.  Use 416 for faster training or\n"
        "1280 for higher precision on large images.",
    )

    _lbl("Epochs  (e.g. 300)")
    epochs_entry = ctk.CTkEntry(
        config_panel, placeholder_text="300", font=FENT, height=36
    )
    epochs_entry.insert(0, "300")
    epochs_entry.pack(fill="x", **PAD)
    Tooltip(
        epochs_entry,
        "Number of full passes over the training dataset.\n"
        "More epochs → longer training, potentially better accuracy.\n"
        "Start with 50–300; increase if validation loss is still improving.",
    )

    _lbl("Batch Size  (e.g. 16)")
    batch_size_entry = ctk.CTkEntry(
        config_panel, placeholder_text="16", font=FENT, height=36
    )
    batch_size_entry.insert(0, "16")
    batch_size_entry.pack(fill="x", **PAD)
    Tooltip(
        batch_size_entry,
        "Images processed per gradient-update step.\n"
        "Reduce (e.g. 8 or 4) if you run out of GPU/CPU memory.\n"
        "Larger batches generally train faster but need more RAM.",
    )

    _lbl("Workers  (e.g. 8)")
    workers_entry = ctk.CTkEntry(
        config_panel, placeholder_text="8", font=FENT, height=36
    )
    workers_entry.insert(0, "8")
    workers_entry.pack(fill="x", **PAD)
    Tooltip(
        workers_entry,
        "Number of CPU worker threads used to load data during training.\n"
        "Higher values can speed up data loading on multi-core machines.\n"
        "Reduce to 0 on Windows if you see multiprocessing errors.",
    )
    _sep()

    # ── Advanced Training Options ──────────────────────────────────────────
    _lbl("⚙  Advanced Training Options")

    def _make_spinbox(parent, initial_val, step=1, is_float=False, width=90):
        """Return (frame, StringVar) for a -/entry/+ spinbox."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        var = ctk.StringVar(value=str(initial_val))
        def _adj(delta):
            try:
                cur = float(var.get()) if is_float else int(float(var.get()))
            except ValueError:
                cur = float(initial_val) if is_float else int(initial_val)
            nv = cur + delta * step
            if is_float:
                # Round to avoid floating-point noise
                nv = round(nv, 10)
                var.set(f"{nv:.4g}")
            else:
                var.set(str(int(nv)))
        ctk.CTkButton(frame, text="−", width=30, height=30, font=("Segoe UI", 14),
                      command=lambda: _adj(-1)).pack(side="left")
        ctk.CTkEntry(frame, textvariable=var, width=width, height=30,
                     font=FENT, justify="center").pack(side="left", padx=2)
        ctk.CTkButton(frame, text="+", width=30, height=30, font=("Segoe UI", 14),
                      command=lambda: _adj(1)).pack(side="left")
        return frame, var

    # Time slider (0 = disabled, 1-24 hours)
    _lbl("Max Training Time (hours, 0 = disabled)")
    _train_time_var = ctk.DoubleVar(value=0.0)
    _time_slider = ctk.CTkSlider(
        config_panel, from_=0, to=24, variable=_train_time_var,
        number_of_steps=24,
    )
    _time_slider.pack(fill="x", **PAD)
    _time_val_lbl = ctk.CTkLabel(config_panel, text="0 hrs (disabled)", font=("Segoe UI", 11), anchor="w")
    _time_val_lbl.pack(padx=14, anchor="w")
    def _on_time_change(*_):
        v = int(_train_time_var.get())
        _time_val_lbl.configure(text=f"{v} hr{'s' if v != 1 else ''}" + (" (disabled)" if v == 0 else ""))
    _train_time_var.trace_add("write", _on_time_change)
    Tooltip(_time_slider,
        "Maximum training time in hours. If set, this overrides the epochs argument,\n"
        "allowing training to automatically stop after the specified duration.\n"
        "Useful for time-constrained training scenarios.\n"
        "Set to 0 to disable (use epochs instead).")

    # Patience spinbox
    _lbl("Patience (early-stop epochs)")
    _patience_row, _train_patience_var = _make_spinbox(config_panel, 100)
    _patience_row.pack(fill="x", **PAD)
    Tooltip(_patience_row,
        "Number of epochs to wait without improvement in validation metrics\n"
        "before early stopping the training. Helps prevent overfitting by\n"
        "stopping training when performance plateaus.")

    # Save toggle and Save Period spinbox on same row
    _save_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _save_row.pack(fill="x", **PAD)
    _train_save_var = ctk.BooleanVar(value=True)
    _save_sw = ctk.CTkSwitch(_save_row, text="Save Checkpoints", variable=_train_save_var, font=("Segoe UI", 12))
    _save_sw.pack(side="left", padx=(0, 16))
    Tooltip(_save_sw,
        "Enables saving of training checkpoints and final model weights.\n"
        "Useful for resuming training or model deployment.")
    _lbl_sp = ctk.CTkLabel(_save_row, text="Save Period:", font=("Segoe UI", 12), anchor="w")
    _lbl_sp.pack(side="left")
    _sp_frame, _train_save_period_var = _make_spinbox(_save_row, -1, width=70)
    _sp_frame.pack(side="left", padx=(4, 0))
    Tooltip(_sp_frame,
        "Frequency of saving model checkpoints (epochs). -1 = disabled.\n"
        "Useful for saving interim models during long training sessions.")

    # Cache toggle
    _cache_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _cache_row.pack(fill="x", **PAD)
    _train_cache_var = ctk.BooleanVar(value=False)
    _cache_sw = ctk.CTkSwitch(_cache_row, text="Cache Dataset Images", variable=_train_cache_var, font=("Segoe UI", 12))
    _cache_sw.pack(side="left")
    Tooltip(_cache_sw,
        "Enables caching of dataset images in memory (True/ram), on disk (disk),\n"
        "or disables it (False). Improves training speed by reducing disk I/O\n"
        "at the cost of increased memory usage.")

    # Freeze spinbox
    _lbl("Freeze Layers (0 = disabled)")
    _freeze_row, _train_freeze_var = _make_spinbox(config_panel, 0)
    _freeze_row.pack(fill="x", **PAD)
    Tooltip(_freeze_row,
        "Freezes the first N layers of the model, reducing the number of\n"
        "trainable parameters. Useful for fine-tuning or transfer learning.\n"
        "Set to 0 to disable.")

    # Learning rates row
    _lr_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _lr_row.pack(fill="x", **PAD)
    ctk.CTkLabel(_lr_row, text="Initial LR (lr0):", font=("Segoe UI", 12), anchor="w").pack(side="left")
    _lr0_frame, _train_lr0_var = _make_spinbox(_lr_row, "0.01", step=0.001, is_float=True, width=70)
    _lr0_frame.pack(side="left", padx=(4, 12))
    Tooltip(_lr0_frame,
        "Initial learning rate (e.g. SGD=1E-2, Adam=1E-3). Adjusting this value\n"
        "is crucial for the optimization process, influencing how rapidly model\n"
        "weights are updated.")
    ctk.CTkLabel(_lr_row, text="Final LR (lrf):", font=("Segoe UI", 12), anchor="w").pack(side="left")
    _lrf_frame, _train_lrf_var = _make_spinbox(_lr_row, "0.01", step=0.001, is_float=True, width=70)
    _lrf_frame.pack(side="left", padx=(4, 0))
    Tooltip(_lrf_frame,
        "Final learning rate as a fraction of the initial rate = (lr0 * lrf),\n"
        "used in conjunction with schedulers to adjust the learning rate over time.")

    # Momentum and Weight Decay row
    _mom_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _mom_row.pack(fill="x", **PAD)
    ctk.CTkLabel(_mom_row, text="Momentum:", font=("Segoe UI", 12), anchor="w").pack(side="left")
    _mom_frame, _train_momentum_var = _make_spinbox(_mom_row, "0.937", step=0.01, is_float=True, width=70)
    _mom_frame.pack(side="left", padx=(4, 12))
    Tooltip(_mom_frame,
        "Momentum factor for SGD or beta1 for Adam optimizers, influencing\n"
        "the incorporation of past gradients in the current update.")
    ctk.CTkLabel(_mom_row, text="Weight Decay:", font=("Segoe UI", 12), anchor="w").pack(side="left")
    _wd_frame, _train_weight_decay_var = _make_spinbox(_mom_row, "0.0005", step=0.0001, is_float=True, width=70)
    _wd_frame.pack(side="left", padx=(4, 0))
    Tooltip(_wd_frame,
        "L2 regularization term, penalizing large weights to prevent overfitting.")

    # Optimizer dropdown
    _lbl("Optimizer")
    _OPTIMIZER_OPTIONS = ["auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]
    _train_optimizer_var = ctk.StringVar(value="auto")
    _opt_menu = ctk.CTkOptionMenu(
        config_panel, values=_OPTIMIZER_OPTIONS, variable=_train_optimizer_var,
        font=FBTN, height=32,
    )
    _opt_menu.pack(fill="x", **PAD)
    Tooltip(_opt_menu,
        "Choice of optimizer for training. Options include SGD, MuSGD, Adam,\n"
        "Adamax, AdamW, NAdam, RAdam, RMSProp, or auto for automatic selection\n"
        "based on model configuration. Affects convergence speed and stability.")

    # Val toggle and Max Det spinbox on same row
    _val_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    _val_row.pack(fill="x", **PAD)
    _train_val_var = ctk.BooleanVar(value=True)
    _val_sw = ctk.CTkSwitch(_val_row, text="Validation", variable=_train_val_var, font=("Segoe UI", 12))
    _val_sw.pack(side="left", padx=(0, 16))
    Tooltip(_val_sw,
        "Enables validation during training, allowing for periodic evaluation\n"
        "of model performance on a separate dataset.")
    _lbl_md = ctk.CTkLabel(_val_row, text="Max Detections:", font=("Segoe UI", 12), anchor="w")
    _lbl_md.pack(side="left")
    _md_frame, _train_max_det_var = _make_spinbox(_val_row, 300, width=70)
    _md_frame.pack(side="left", padx=(4, 0))
    _train_max_det_widget = _md_frame
    Tooltip(_md_frame,
        "Specifies the maximum number of objects retained during the validation\n"
        "phase of training. Only applicable when Validation is enabled.")
    def _on_val_change(*_):
        state = "normal" if _train_val_var.get() else "disabled"
        for child in _md_frame.winfo_children():
            try:
                child.configure(state=state)
            except Exception:
                pass
    _train_val_var.trace_add("write", _on_val_change)
    _sep()

    # ── Class names ────────────────────────────────────────────────────────
    _lbl("Class Names  (one per line)")
    class_names_text = ctk.CTkTextbox(config_panel, font=FENT, height=110)
    class_names_text.pack(fill="x", **PAD)
    Tooltip(
        class_names_text,
        "Enter each object class on its own line, matching the class IDs\n"
        "in your annotation .txt files (line 1 = class 0, etc.).\n\n"
        "Example:\n"
        "  cat\n"
        "  dog\n"
        "  car\n\n"
        "If you imported a Roboflow ZIP or selected a folder with data.yaml,\n"
        "these are filled automatically.",
    )

    # Store reference so select_train_data() can auto-fill it
    global _train_class_names_text
    _train_class_names_text = class_names_text
    _sep()

    # ── Training queue ─────────────────────────────────────────────────────
    _lbl("Training Queue")

    def _collect_extra_params(workers_str="8"):
        """Collect the advanced training params from the widget variables."""
        ep = {}
        ep['workers'] = int(workers_str) if str(workers_str).isdigit() else 8
        # time
        try:
            ep['time'] = int(_train_time_var.get()) if _train_time_var else 0
        except Exception:
            ep['time'] = 0
        # patience
        try:
            ep['patience'] = int(_train_patience_var.get()) if _train_patience_var else 100
        except Exception:
            ep['patience'] = 100
        ep['save'] = bool(_train_save_var.get()) if _train_save_var else True
        try:
            ep['save_period'] = int(_train_save_period_var.get()) if _train_save_period_var else -1
        except Exception:
            ep['save_period'] = -1
        ep['cache'] = bool(_train_cache_var.get()) if _train_cache_var else False
        ep['resume'] = bool(_train_resume_var.get()) if _train_resume_var else False
        try:
            ep['freeze'] = int(_train_freeze_var.get()) if _train_freeze_var else 0
        except Exception:
            ep['freeze'] = 0
        try:
            ep['lr0'] = float(_train_lr0_var.get()) if _train_lr0_var else 0.01
        except Exception:
            ep['lr0'] = 0.01
        try:
            ep['lrf'] = float(_train_lrf_var.get()) if _train_lrf_var else 0.01
        except Exception:
            ep['lrf'] = 0.01
        try:
            ep['momentum'] = float(_train_momentum_var.get()) if _train_momentum_var else 0.937
        except Exception:
            ep['momentum'] = 0.937
        try:
            ep['weight_decay'] = float(_train_weight_decay_var.get()) if _train_weight_decay_var else 0.0005
        except Exception:
            ep['weight_decay'] = 0.0005
        ep['optimizer'] = str(_train_optimizer_var.get()) if _train_optimizer_var else 'auto'
        ep['val'] = bool(_train_val_var.get()) if _train_val_var else True
        try:
            ep['max_det'] = int(_train_max_det_var.get()) if _train_max_det_var else 300
        except Exception:
            ep['max_det'] = 300
        return ep

    def _get_current_job_config():
        pname   = project_name_entry.get().strip()
        isize   = input_size_entry.get().strip() or "640"
        ep      = epochs_entry.get().strip() or "300"
        bs      = batch_size_entry.get().strip() or "16"
        wk      = workers_entry.get().strip() or "8"
        raw_cls = class_names_text.get("1.0", "end-1c")
        cls     = [n.strip() for n in raw_cls.splitlines() if n.strip()]
        sel_disp  = selected_model_var.get() if selected_model_var else ""
        sel_model = MODEL_MAP.get(sel_disp, "")
        extra = _collect_extra_params(wk)
        return {
            "project_name":      pname,
            "input_size":        isize,
            "epochs":            ep,
            "batch_size":        bs,
            "workers":           wk,
            "class_names":       cls,
            "selected_model":    sel_model,
            "custom_model_path": custom_model_path,
            "train_data_path":   train_data_path,
            "model_save_path":   model_save_path,
            "roboflow_yaml":     roboflow_yaml_path,
            "extra_params":      extra,
        }

    def _refresh_queue_list():
        global _train_queue_frame
        if _train_queue_frame is None:
            return
        for w in _train_queue_frame.winfo_children():
            w.destroy()
        if not _train_queue:
            ctk.CTkLabel(
                _train_queue_frame, text="No jobs queued.", font=("Segoe UI", 11),
                text_color="gray",
            ).pack(padx=8, pady=4)
            return
        for qi, job in enumerate(_train_queue):
            row = ctk.CTkFrame(_train_queue_frame, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=1)
            lbl_txt = f"{qi + 1}. {job['project_name'] or '(unnamed)'}  [{job['epochs']} ep]"
            ctk.CTkLabel(row, text=lbl_txt, font=("Segoe UI", 11), anchor="w").pack(
                side="left", fill="x", expand=True, padx=4
            )
            ctk.CTkButton(
                row, text="✕", width=28, height=22,
                fg_color="#c62828", hover_color="#b71c1c",
                font=("Segoe UI", 10), text_color="white",
                command=lambda x=qi: (_train_queue.pop(x), _refresh_queue_list()),
            ).pack(side="right", padx=2)

    global _train_queue_frame
    _train_queue_frame = ctk.CTkScrollableFrame(
        config_panel, height=90, fg_color="#2a2a3e", corner_radius=6
    )
    _train_queue_frame.pack(fill="x", padx=14, pady=4)
    _refresh_queue_list()

    q_btns = ctk.CTkFrame(config_panel, fg_color="transparent")
    q_btns.pack(fill="x", padx=14, pady=(2, 4))

    def _add_to_queue():
        cfg = _get_current_job_config()
        if not cfg["project_name"]:
            messagebox.showerror("Queue", "Enter a Project Name before adding to queue.")
            return
        _train_queue.append(cfg)
        _refresh_queue_list()
        output_queue.put(f"✅ Added '{cfg['project_name']}' to queue ({len(_train_queue)} jobs)\n")

    def _clear_queue():
        _train_queue.clear()
        _refresh_queue_list()

    ctk.CTkButton(
        q_btns, text="➕ Add to Queue", font=("Segoe UI", 12), height=30,
        fg_color="#1565c0", hover_color="#0d47a1",
        command=_add_to_queue,
    ).pack(side="left", expand=True, fill="x", padx=(0, 4))
    Tooltip(
        q_btns.winfo_children()[0],
        "Snapshot the current settings and add them as a queued training job.\n"
        "Use 'Run Queue' to execute all queued jobs one after another.",
    )
    ctk.CTkButton(
        q_btns, text="Clear Queue", font=("Segoe UI", 12), height=30,
        fg_color="gray50", hover_color="gray35",
        command=_clear_queue,
    ).pack(side="left", expand=True, fill="x", padx=(4, 0))
    _sep()

    # ── Start / Stop / Queue-Run buttons ──────────────────────────────────
    btn_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    btn_row.pack(fill="x", padx=14, pady=(4, 4))

    ctk.CTkButton(
        btn_row,
        text="▶  Start Training",
        command=lambda: start_training(
            project_name_entry, input_size_entry, epochs_entry,
            batch_size_entry, class_names_text, workers_entry,
        ),
        fg_color="#2e7d32",
        hover_color="#1b5e20",
        font=("Segoe UI", 14, "bold"),
        height=46,
        text_color="white",
        corner_radius=8,
    ).pack(side="left", fill="x", expand=True, padx=(0, 4))
    Tooltip(
        btn_row.winfo_children()[0],
        "Start training immediately with the current settings.",
    )

    _stop_btn = ctk.CTkButton(
        btn_row,
        text="⏹  Stop Training",
        command=_stop_training,
        fg_color="#b71c1c",
        hover_color="#7f0000",
        font=("Segoe UI", 13, "bold"),
        height=46,
        text_color="white",
        corner_radius=8,
        state="disabled",
    )
    _stop_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
    Tooltip(
        _stop_btn,
        "Stop the running training process.\n"
        "The current epoch will be allowed to finish before stopping.",
    )
    _train_stop_btn_ref[0] = _stop_btn

    stop_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    stop_row.pack(fill="x", padx=14, pady=(0, 4))

    ctk.CTkButton(
        stop_row,
        text="▶▶  Run Queue",
        command=_run_training_queue,
        fg_color="#4a148c",
        hover_color="#311b92",
        font=("Segoe UI", 14, "bold"),
        height=46,
        text_color="white",
        corner_radius=8,
    ).pack(fill="x")
    Tooltip(
        stop_row.winfo_children()[0],
        "Run all queued training jobs in order.\n"
        "Each job uses the settings it was queued with.",
    )

    # ── Log panel ──────────────────────────────────────────────────────────
    ctk.CTkLabel(
        log_panel, text="Training Output", font=("Segoe UI", 14, "bold")
    ).pack(anchor="w", padx=12, pady=(10, 4))

    output_textbox = ctk.CTkTextbox(
        log_panel, font=("Courier New", 12), corner_radius=8
    )
    output_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 4))

    # Restore any training log that was produced before this tab was loaded
    if _train_log_buffer:
        output_textbox.insert("1.0", "".join(_train_log_buffer))
        output_textbox.yview_moveto(1)

    train_progress_label = ctk.CTkLabel(
        log_panel, text=_train_progress_text, font=("Segoe UI", 11), text_color="#a6adc8", anchor="w",
    )
    train_progress_label.pack(anchor="w", padx=12)

    progress_bar = ctk.CTkProgressBar(
        log_panel, progress_color="#43a047", mode="determinate",
    )
    progress_bar.set(_train_progress_value)
    progress_bar.pack(fill="x", padx=12, pady=(2, 10))
    # Store label reference on progress_bar object for access in callback
    progress_bar._progress_label = train_progress_label


# ─────────────────────────────────────────────────────────────────────────────
#  Image / Video Detection window  (redesigned)
# ─────────────────────────────────────────────────────────────────────────────
def show_image_detection_window() -> None:
    global image_label, detection_progress_bar, image_index_label
    global detect_folder_label, detect_model_label
    global _detect_start_btn, _detect_controls
    global _detect_file_count_label, _detect_model_info_label
    global _detect_conf_var, _detect_half_var, _detect_workers_var, _detect_task_var
    global _detect_progress_label, _detect_nav_bar

    FONT  = ("Segoe UI", 12)
    FLAB  = ("Segoe UI", 12)
    FBTN  = ("Segoe UI", 12)

    # ── Left config panel ─────────────────────────────────────────────────
    cfg = ctk.CTkScrollableFrame(
        main_frame, label_text="Detection Configuration",
        label_font=("Segoe UI", 13, "bold"),
        corner_radius=8, width=260,
    )
    cfg.place(relx=0, rely=0, relwidth=0.28, relheight=1.0)

    def _clbl(text):
        l = ctk.CTkLabel(cfg, text=text, font=FLAB, anchor="w")
        l.pack(fill="x", padx=12, pady=(8, 1))
        return l

    def _csep():
        ctk.CTkFrame(cfg, height=1, fg_color="gray50").pack(fill="x", padx=12, pady=3)

    # Folder section
    _clbl("📁  Images / Videos Folder")
    sel_folder_btn = ctk.CTkButton(
        cfg, text="Browse Folder…", font=FBTN, height=34,
        command=select_detection_images_folder,
    )
    sel_folder_btn.pack(fill="x", padx=12, pady=(4, 2))
    Tooltip(sel_folder_btn, "Select a folder containing images or videos to run detection on.")

    detect_folder_label = ctk.CTkLabel(
        cfg, text=Path(detection_images_folder_path).name if detection_images_folder_path else "No folder selected",
        font=("Segoe UI", 10),
        text_color="#4caf50" if detection_images_folder_path else "gray",
        anchor="w", wraplength=220,
    )
    detect_folder_label.pack(fill="x", padx=12)

    _detect_file_count_label = ctk.CTkLabel(
        cfg, text=_detect_file_count_text, font=("Segoe UI", 10), text_color="#6c7086", anchor="w",
    )
    _detect_file_count_label.pack(fill="x", padx=12)
    _csep()

    # Model section
    _clbl("🤖  YOLO Model (.pt / .engine / .onnx)")
    sel_model_btn = ctk.CTkButton(
        cfg, text="Browse Model…", font=FBTN, height=34,
        command=select_detection_model,
    )
    sel_model_btn.pack(fill="x", padx=12, pady=(4, 2))
    Tooltip(sel_model_btn, "Choose a trained YOLO weights file for inference.\nSupports .pt, TensorRT .engine, ONNX .onnx, and other exported formats.")

    detect_model_label = ctk.CTkLabel(
        cfg, text=Path(detection_model_path).name if detection_model_path else "No model selected",
        font=("Segoe UI", 10),
        text_color="#4caf50" if detection_model_path else "gray",
        anchor="w", wraplength=220,
    )
    detect_model_label.pack(fill="x", padx=12)

    _detect_model_info_label = ctk.CTkLabel(
        cfg, text=_detect_model_info_text, font=("Segoe UI", 10), text_color="#6c7086",
        anchor="w", wraplength=220, justify="left",
    )
    _detect_model_info_label.pack(fill="x", padx=12)
    _csep()

    # Confidence threshold
    _clbl("Confidence Threshold")
    _detect_conf_var = ctk.DoubleVar(value=0.5)
    conf_slider = ctk.CTkSlider(
        cfg, from_=0.01, to=1.0, variable=_detect_conf_var,
        number_of_steps=99,
    )
    conf_slider.pack(fill="x", padx=12, pady=(4, 0))
    conf_val_lbl = ctk.CTkLabel(cfg, text="0.50", font=("Segoe UI", 11), anchor="w")
    conf_val_lbl.pack(padx=12, anchor="w")
    _detect_conf_var.trace_add(
        "write",
        lambda *_: conf_val_lbl.configure(text=f"{_detect_conf_var.get():.2f}"),
    )
    Tooltip(
        conf_slider,
        "Only detections with confidence ≥ this value will be shown.\n"
        "0.5 = 50% certainty required.  Lower → more detections (more false positives).\n"
        "Higher → fewer but more reliable detections.",
    )
    _csep()

    # Half-precision (FP16) option
    _detect_half_var = ctk.BooleanVar(value=False)
    half_chk = ctk.CTkCheckBox(
        cfg, text="Half Precision (FP16)", variable=_detect_half_var, font=FLAB,
    )
    half_chk.pack(fill="x", padx=12, pady=(6, 2))
    Tooltip(
        half_chk,
        "Run inference in 16-bit floating-point mode (FP16).\n"
        "Roughly 2× faster on NVIDIA GPUs with Tensor Cores, using half the GPU memory.\n"
        "Requires a CUDA-capable GPU; ignored on CPU.",
    )

    # Workers
    _clbl("Data Workers")
    _detect_workers_var = ctk.StringVar(value="4")
    workers_entry_d = ctk.CTkEntry(cfg, textvariable=_detect_workers_var, font=FLAB, height=32)
    workers_entry_d.pack(fill="x", padx=12, pady=(2, 4))
    Tooltip(
        workers_entry_d,
        "Number of CPU threads used to load images in parallel.\n"
        "Increase for faster throughput on multi-core machines.\n"
        "Set to 0 on Windows if you see multiprocessing errors.",
    )
    _csep()

    # Task override (required for TensorRT / ONNX exported models)
    _TASK_OPTIONS = ["Auto-detect", "detect", "segment", "classify", "pose", "obb"]
    _clbl("Model Task")
    _detect_task_var = ctk.StringVar(value=_TASK_OPTIONS[0])
    task_menu = ctk.CTkOptionMenu(
        cfg, values=_TASK_OPTIONS, variable=_detect_task_var, font=FLAB, height=32,
    )
    task_menu.pack(fill="x", padx=12, pady=(2, 4))
    Tooltip(
        task_menu,
        "Task type for the YOLO model.\n"
        "'Auto-detect' works for .pt files but will fail for exported\n"
        "formats like TensorRT (.engine) or ONNX (.onnx) that do not\n"
        "embed task metadata.  In those cases select the correct task\n"
        "explicitly (e.g. 'segment' for segmentation models).",
    )
    _csep()

    # Start / Cancel button
    _detect_start_btn = ctk.CTkButton(
        cfg, text="▶  Start Detection",
        command=toggle_image_detection,
        fg_color="#1565c0", hover_color="#0d47a1",
        font=("Segoe UI", 14, "bold"), height=46,
        text_color="white", corner_radius=8,
    )
    _detect_start_btn.pack(fill="x", padx=12, pady=(8, 4))

    # Gallery button (enabled after results exist)
    gallery_btn = ctk.CTkButton(
        cfg, text="🖼  Open Gallery",
        command=_open_gallery,
        fg_color="#4a148c", hover_color="#311b92",
        font=("Segoe UI", 13, "bold"), height=36,
        text_color="white", corner_radius=8,
    )
    gallery_btn.pack(fill="x", padx=12, pady=(0, 8))
    Tooltip(gallery_btn, "View all detection result images in a scrollable thumbnail gallery.\nClick any thumbnail to open it full-screen.")

    _detect_controls = [sel_folder_btn, sel_model_btn, conf_slider, half_chk,
                        workers_entry_d, task_menu]

    # ── Right: image viewer ────────────────────────────────────────────────
    viewer = ctk.CTkFrame(main_frame, corner_radius=8, fg_color="#0d1117")
    viewer.place(relx=0.29, rely=0, relwidth=0.71, relheight=1.0)

    image_label = tk.Label(viewer, bg="#0d1117")
    image_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.88)

    # Bottom navigation bar (compact, two slim rows)
    _detect_nav_bar = ctk.CTkFrame(viewer, corner_radius=0, fg_color="#1e1e2e")
    _detect_nav_bar.place(relx=0, rely=0.88, relwidth=1.0, relheight=0.12)

    # ── Row 1: navigation controls (top half of bar) ─────────────────────
    ctk.CTkButton(
        _detect_nav_bar, text="◀", command=show_prev_image,
        fg_color="#1976d2", font=("Segoe UI", 18, "bold"), height=32, width=40,
    ).place(relx=0.01, rely=0.05, relwidth=0.07, relheight=0.42)

    image_index_label = ctk.CTkLabel(_detect_nav_bar, text="No results", font=("Segoe UI", 11))
    image_index_label.place(relx=0.09, rely=0.05, relwidth=0.18, relheight=0.42)

    ctk.CTkButton(
        _detect_nav_bar, text="▶", command=show_next_image,
        fg_color="#1976d2", font=("Segoe UI", 18, "bold"), height=32, width=40,
    ).place(relx=0.28, rely=0.05, relwidth=0.07, relheight=0.42)

    ctk.CTkButton(
        _detect_nav_bar, text="⛶  Full Screen", command=_open_fullscreen_image,
        fg_color="#37474f", hover_color="#263238",
        font=("Segoe UI", 11), height=28,
    ).place(relx=0.37, rely=0.05, relwidth=0.14, relheight=0.42)
    Tooltip(_detect_nav_bar.winfo_children()[-1], "Open the current image in a full-screen viewer.")

    # ── Zoom label (top row, right of Full Screen) ────────────────────────
    global _detect_zoom_var
    _detect_zoom_var = ctk.DoubleVar(value=1.0)

    zoom_lbl = ctk.CTkLabel(
        _detect_nav_bar, text="🔍 1.00×", font=("Segoe UI", 9), anchor="w"
    )
    zoom_lbl.place(relx=0.52, rely=0.05, relwidth=0.055, relheight=0.42)

    # ── Zoom slider (top row, right of zoom label) ─────────────────────────
    zoom_slider = ctk.CTkSlider(
        _detect_nav_bar, from_=0.25, to=2.0,
        variable=_detect_zoom_var,
        number_of_steps=int((2.0 - 0.25) / 0.05),  # 0.05× steps
        height=14,
    )
    zoom_slider.place(relx=0.578, rely=0.10, relwidth=0.075, relheight=0.32)

    # ── Progress label (top row, right of zoom slider) ─────────────────────
    _detect_progress_label = ctk.CTkLabel(
        _detect_nav_bar, text=_detect_progress_text,
        font=("Segoe UI", 9), text_color="#a6adc8", anchor="w",
    )
    _detect_progress_label.place(relx=0.66, rely=0.05, relwidth=0.33, relheight=0.42)

    def _on_zoom(*_):
        z = _detect_zoom_var.get()
        zoom_lbl.configure(text=f"🔍 {z:.2f}×")
        if image_paths:
            update_image()

    _detect_zoom_var.trace_add("write", _on_zoom)
    Tooltip(
        zoom_slider,
        "Adjust the display size of the result image.\n"
        "0.25× = thumbnail view   1.00× = fit to panel   2.00× = zoomed in",
    )

    # ── Progress bar (thin full-width strip at bottom of nav bar) ──────────
    detection_progress_bar = ctk.CTkProgressBar(
        _detect_nav_bar, progress_color="#43a047", mode="determinate", height=5,
    )
    detection_progress_bar.set(_detect_progress_value)
    detection_progress_bar.place(relx=0.01, rely=0.62, relwidth=0.98, relheight=0.20)

    # Restore previously loaded result images if any exist
    if image_paths:
        root.after(150, update_image)


# ─────────────────────────────────────────────────────────────────────────────
#  Camera Detection window
# ─────────────────────────────────────────────────────────────────────────────
def show_camera_detection_window() -> None:
    global camera_detection, camera_id_entry, image_label, _camera_bar, _camera_half_var

    camera_detection = None

    image_label = tk.Label(main_frame, bg="#0d0d0d")
    image_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.93)

    bar = ctk.CTkFrame(main_frame, corner_radius=0, height=50)
    bar.place(relx=0, rely=0.93, relwidth=1.0, relheight=0.07)
    _camera_bar = bar

    FONT = ("Segoe UI", 12)

    sel_model_btn = ctk.CTkButton(
        bar, text="Select Model (.pt)",
        command=select_detection_model, font=FONT, height=34,
    )
    sel_model_btn.place(relx=0.01, rely=0.1, relwidth=0.13, relheight=0.8)
    Tooltip(sel_model_btn, "Choose the YOLO .pt model file used for live camera inference.")

    sel_save_btn = ctk.CTkButton(
        bar, text="Save Folder",
        command=select_camera_save_folder, font=FONT, height=34,
    )
    sel_save_btn.place(relx=0.16, rely=0.1, relwidth=0.10, relheight=0.8)
    Tooltip(sel_save_btn, "Folder where captured frames are saved when you press Enter.")

    camera_id_entry = ctk.CTkEntry(
        bar, placeholder_text="Camera ID  (e.g. 0)", font=FONT, height=34
    )
    camera_id_entry.place(relx=0.28, rely=0.1, relwidth=0.14, relheight=0.8)
    Tooltip(
        camera_id_entry,
        "Index of the camera to open.\n"
        "0 = default webcam, 1 = second camera, etc.",
    )

    # Half-precision (FP16) toggle
    _camera_half_var = ctk.BooleanVar(value=False)
    half_chk = ctk.CTkCheckBox(
        bar, text="FP16", variable=_camera_half_var, font=FONT,
    )
    half_chk.place(relx=0.44, rely=0.15, relwidth=0.07, relheight=0.70)
    Tooltip(
        half_chk,
        "Run camera inference in half-precision (FP16) mode.\n"
        "Approximately 2× faster on NVIDIA GPUs with Tensor Cores.\n"
        "Requires a CUDA-capable GPU; ignored on CPU.",
    )

    ctk.CTkLabel(
        bar,
        text="Press  Enter  to capture & save a frame",
        font=("Segoe UI", 11),
        text_color="gray",
    ).place(relx=0.53, rely=0.1, relwidth=0.25, relheight=0.8)

    start_cam_btn = ctk.CTkButton(
        bar, text="▶  START",
        command=start_camera_detection,
        fg_color="#2e7d32", hover_color="#1b5e20",
        font=("Segoe UI", 14, "bold"), height=34, text_color="white",
    )
    start_cam_btn.place(relx=0.80, rely=0.1, relwidth=0.18, relheight=0.8)
    bar._start_btn = start_cam_btn

    root.bind("<Return>", lambda _e: save_callback())
    image_label.update_idletasks()


# ─────────────────────────────────────────────────────────────────────────────
#  Live Video Detection window
# ─────────────────────────────────────────────────────────────────────────────
def show_live_video_window() -> None:
    global _live_video_path, _live_video_label, _live_video_status_label
    global _live_video_start_btn, _live_video_bar, detection_model_path
    global _live_video_pause_btn, _live_video_seek_slider
    global _live_video_half_var, _live_video_conf_var, _live_video_task_var
    global _live_audio_enabled_var, _live_audio_sync_var, _live_audio_volume_var

    _live_video_path = ""
    _live_video_cancel_flag[0] = False
    _live_video_is_url[0] = False

    # ── Display area ──────────────────────────────────────────────────────
    _live_video_label = tk.Label(main_frame, bg="#0d0d0d")
    _live_video_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.78)
    # Left-click on the playback window toggles play/pause (only when playing)
    _live_video_label.bind("<Button-1>", lambda _e: _toggle_pause() if _live_video_running else None)

    # ── Seek / position slider strip ──────────────────────────────────────
    seek_strip = ctk.CTkFrame(main_frame, corner_radius=0, fg_color="#111111", height=28)
    seek_strip.place(relx=0, rely=0.78, relwidth=1.0, relheight=0.04)

    def _on_seek(val: str) -> None:
        total = _live_video_total_ref[0]
        target = int(float(val) * max(total - 1, 1))
        _live_video_seek_to[0] = target
        _live_video_seeking[0] = True
        # Hold the "seeking" flag briefly so the thread doesn't immediately
        # overwrite the slider position before the seek has been processed.
        root.after(400, lambda: _live_video_seeking.__setitem__(0, False))

    _live_video_seek_slider = ctk.CTkSlider(
        seek_strip, from_=0.0, to=1.0, command=_on_seek,
        height=20, progress_color="#4caf50", button_color="#a6e3a1",
    )
    _live_video_seek_slider.set(0)
    _live_video_seek_slider.place(relx=0.01, rely=0.1, relwidth=0.98, relheight=0.8)

    # ── Controls bar ──────────────────────────────────────────────────────
    bar = ctk.CTkFrame(main_frame, corner_radius=0, fg_color="#1e1e2e")
    bar.place(relx=0, rely=0.82, relwidth=1.0, relheight=0.18)
    _live_video_bar = bar

    FONT = ("Segoe UI", 11)

    # Info labels (top row)
    _video_path_lbl = ctk.CTkLabel(
        bar, text="No video selected", font=FONT, text_color="gray", anchor="w",
    )
    _video_path_lbl.place(relx=0.01, rely=0.04, relwidth=0.48, relheight=0.28)

    _live_video_status_label = ctk.CTkLabel(
        bar, text="", font=FONT, text_color="#a6adc8", anchor="w",
    )
    _live_video_status_label.place(relx=0.01, rely=0.34, relwidth=0.48, relheight=0.28)

    # ── Half precision (FP16) ──────────────────────────────────────────────
    _live_video_half_var = ctk.BooleanVar(value=False)
    half_chk = ctk.CTkCheckBox(
        bar, text="FP16", variable=_live_video_half_var, font=FONT,
    )
    half_chk.place(relx=0.16, rely=0.56, relwidth=0.07, relheight=0.38)
    Tooltip(half_chk, "Run inference in half-precision (FP16) for ~2× speed on NVIDIA GPUs.")

    # ── Confidence threshold (compact) ────────────────────────────────────
    _live_video_conf_var = ctk.DoubleVar(value=0.5)
    conf_lbl = ctk.CTkLabel(bar, text="Conf:", font=FONT, anchor="w")
    conf_lbl.place(relx=0.24, rely=0.56, relwidth=0.03, relheight=0.38)
    conf_slider = ctk.CTkSlider(
        bar, from_=0.01, to=1.0, variable=_live_video_conf_var,
        number_of_steps=99, width=70,
    )
    conf_slider.place(relx=0.27, rely=0.64, relwidth=0.07, relheight=0.22)
    conf_val_lbl = ctk.CTkLabel(bar, text="0.50", font=FONT, anchor="w")
    conf_val_lbl.place(relx=0.35, rely=0.56, relwidth=0.04, relheight=0.38)
    _live_video_conf_var.trace_add(
        "write",
        lambda *_: conf_val_lbl.configure(text=f"{_live_video_conf_var.get():.2f}"),
    )
    Tooltip(conf_slider, "Minimum confidence for detections to be shown.")

    # ── Audio controls ─────────────────────────────────────────────────────
    _live_audio_enabled_var = ctk.BooleanVar(value=False)
    audio_chk = ctk.CTkCheckBox(
        bar, text="🔊 Audio", variable=_live_audio_enabled_var, font=FONT,
    )
    audio_chk.place(relx=0.40, rely=0.56, relwidth=0.085, relheight=0.38)
    Tooltip(
        audio_chk,
        "Enable audio playback alongside the detection video.\n\n"
        "Requires ffmpeg to be installed and available on your system PATH.\n"
        "Audio is extracted from the video file at playback start.\n"
        "Not available for URL/stream sources.",
    )

    _live_audio_sync_var = ctk.BooleanVar(value=True)
    sync_chk = ctk.CTkCheckBox(
        bar, text="Sync", variable=_live_audio_sync_var, font=FONT,
    )
    sync_chk.place(relx=0.49, rely=0.56, relwidth=0.065, relheight=0.38)
    Tooltip(
        sync_chk,
        "Continuously adjust audio playback speed to match the actual\n"
        "video frame rate (which may be slower than real-time when YOLO\n"
        "inference is heavy).  Audio slows down smoothly — no pauses or\n"
        "jarring cuts.  Accumulated drift is closed within ~2 frames.\n"
        "Re-enabling Sync seeks audio to the current frame position.\n"
        "Disable to let audio free-run at 1.0× speed.",
    )

    # ── Volume control (placed to the right of the +10s seek button) ──────
    _live_audio_volume_var = ctk.DoubleVar(value=1.0)
    vol_lbl = ctk.CTkLabel(bar, text="Vol:", font=FONT, anchor="w")
    vol_lbl.place(relx=0.82, rely=0.56, relwidth=0.03, relheight=0.38)
    vol_slider = ctk.CTkSlider(
        bar, from_=0.0, to=1.0, variable=_live_audio_volume_var,
        number_of_steps=20,
    )
    vol_slider.place(relx=0.85, rely=0.64, relwidth=0.07, relheight=0.22)
    vol_val_lbl = ctk.CTkLabel(bar, text="100%", font=FONT, anchor="w")
    vol_val_lbl.place(relx=0.93, rely=0.56, relwidth=0.045, relheight=0.38)

    def _on_volume_change(*_):
        v = _live_audio_volume_var.get()
        vol_val_lbl.configure(text=f"{int(v * 100)}%")
        _pcm_set_volume(v)

    _live_audio_volume_var.trace_add("write", _on_volume_change)
    Tooltip(vol_slider, "Audio playback volume (0 – 100%).")

    # ── Buttons (right half) ───────────────────────────────────────────────
    def _pick_video():
        global _live_video_path
        p = normalize_path(
            filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("All files", "*.*"),
                ],
            )
        )
        if p:
            _live_video_path = p
            _live_video_is_url[0] = False
            _video_path_lbl.configure(text=Path(p).name, text_color="#4caf50")

    def _pick_url():
        """Prompt the user for a video URL and use it as the source."""
        global _live_video_path
        win = tk.Toplevel(root)
        win.title("Open Video URL")
        win.geometry("480x140")
        win.resizable(False, False)
        win.configure(bg="#1e1e2e")
        win.grab_set()
        ctk.CTkLabel(
            win,
            text="Enter video or stream URL:",
            font=("Segoe UI", 12),
        ).pack(pady=(16, 4), padx=20, anchor="w")
        url_entry = ctk.CTkEntry(win, font=("Segoe UI", 12), height=36, width=440)
        url_entry.pack(padx=20, pady=(0, 10))
        url_entry.focus_set()

        def _confirm():
            url = url_entry.get().strip()
            if url:
                _live_video_path = url
                _live_video_is_url[0] = True
                short = url if len(url) <= 50 else url[:47] + "…"
                _video_path_lbl.configure(text=short, text_color="#89b4fa")
            win.destroy()

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=20)
        ctk.CTkButton(
            btn_row, text="Open", fg_color="#1565c0",
            font=("Segoe UI", 12), height=32,
            command=_confirm,
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(
            btn_row, text="Cancel", fg_color="#37474f",
            font=("Segoe UI", 12), height=32,
            command=win.destroy,
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))
        win.bind("<Return>", lambda _e: _confirm())

    def _pick_model():
        global detection_model_path
        p = normalize_path(
            filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[
                    ("YOLO model", "*.pt *.onnx *.engine"),
                    ("All files", "*.*"),
                ],
            )
        )
        if p:
            detection_model_path = p
            _safe_label_configure(
                _live_video_status_label,
                text=f"Model: {Path(p).name}",
                text_color="#64b5f6",
            )

    vid_btn = ctk.CTkButton(bar, text="📂 Video", command=_pick_video, font=FONT, height=28)
    vid_btn.place(relx=0.51, rely=0.06, relwidth=0.08, relheight=0.38)
    Tooltip(vid_btn, "Choose a local video file for detection playback.")

    url_btn = ctk.CTkButton(
        bar, text="🌐 URL", command=_pick_url,
        fg_color="#37474f", hover_color="#263238",
        font=FONT, height=28,
    )
    url_btn.place(relx=0.60, rely=0.06, relwidth=0.07, relheight=0.38)
    Tooltip(
        url_btn,
        "Open a video or stream from a URL.\n\n"
        "Supported schemes: http://, https://, rtsp://, rtp://, udp://\n\n"
        "Note: audio playback is not available for URL sources.",
    )

    model_btn = ctk.CTkButton(bar, text="🤖 Model", command=_pick_model, font=FONT, height=28)
    model_btn.place(relx=0.68, rely=0.06, relwidth=0.08, relheight=0.38)
    Tooltip(model_btn, "Choose a YOLO model (.pt, .onnx, or .engine) for detection.")

    def _toggle_pause():
        global _live_video_paused
        _live_video_paused = not _live_video_paused
        if _live_video_pause_btn:
            _live_video_pause_btn.configure(
                text="▶ Resume" if _live_video_paused else "⏸ Pause",
                fg_color="#e65100" if _live_video_paused else "#37474f",
            )
        # Pause / unpause audio in sync – set the flag that the callback reads
        if _live_video_paused:
            _pcm_paused[0] = True
        else:
            _pcm_paused[0] = False

    _live_video_pause_btn = ctk.CTkButton(
        bar, text="⏸ Pause", command=_toggle_pause,
        fg_color="#37474f", hover_color="#263238",
        font=FONT, height=28,
    )
    _live_video_pause_btn.place(relx=0.77, rely=0.06, relwidth=0.08, relheight=0.38)
    Tooltip(_live_video_pause_btn, "Pause or resume video playback.  You can also left-click the video image.")

    def _screenshot_dialog():
        if _live_video_raw_frame[0] is None:
            messagebox.showinfo("Screenshot", "No frame available yet — start playback first.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        win = tk.Toplevel(root)
        win.title("Save Screenshot")
        win.geometry("300x130")
        win.resizable(False, False)
        win.configure(bg="#1e1e2e")
        win.grab_set()
        ctk.CTkLabel(win, text="Save screenshot as:", font=("Segoe UI", 12)).pack(pady=(14, 6))
        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=20)

        def _save(with_boxes: bool):
            win.destroy()
            frame = _live_video_ann_frame[0] if with_boxes else _live_video_raw_frame[0]
            if frame is None:
                messagebox.showinfo("Screenshot", "Frame no longer available.")
                return
            suffix = "_detection" if with_boxes else "_raw"
            fname = f"screenshot{suffix}_{ts}.png"
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
                initialfile=fname,
            )
            if path:
                cv2.imwrite(path, frame)
                messagebox.showinfo("Screenshot saved", f"Saved to:\n{path}")

        ctk.CTkButton(
            btn_row, text="With Detections", fg_color="#1565c0",
            font=("Segoe UI", 11), height=32,
            command=lambda: _save(True),
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ctk.CTkButton(
            btn_row, text="Raw Frame", fg_color="#37474f",
            font=("Segoe UI", 11), height=32,
            command=lambda: _save(False),
        ).pack(side="left", expand=True, fill="x", padx=(4, 0))

    shot_btn = ctk.CTkButton(
        bar, text="📷 Screenshot", command=_screenshot_dialog,
        fg_color="#37474f", hover_color="#263238",
        font=FONT, height=28,
    )
    shot_btn.place(relx=0.86, rely=0.06, relwidth=0.06, relheight=0.38)
    Tooltip(shot_btn, "Save the current frame as a PNG — choose with or without detection boxes.")

    def _toggle():
        global _live_video_running
        if _live_video_running:
            _stop_live_video()
        else:
            _start_live_video()

    _live_video_start_btn = ctk.CTkButton(
        bar, text="▶  PLAY",
        command=_toggle,
        fg_color="#2e7d32", hover_color="#1b5e20",
        font=("Segoe UI", 13, "bold"), height=28, text_color="white",
    )
    _live_video_start_btn.place(relx=0.93, rely=0.06, relwidth=0.06, relheight=0.38)
    bar._start_btn = _live_video_start_btn

    # Second row: Task | FP16 | Conf | Audio | Sync | Seek buttons | Volume
    # ── Task override (required for TensorRT / ONNX exported models) ──────
    _LIVE_TASK_OPTIONS = ["Auto-detect", "detect", "segment", "classify", "pose", "obb"]
    _live_video_task_var = ctk.StringVar(value=_LIVE_TASK_OPTIONS[0])
    task_lbl = ctk.CTkLabel(bar, text="Task:", font=FONT, anchor="w")
    task_lbl.place(relx=0.01, rely=0.56, relwidth=0.04, relheight=0.38)
    task_menu = ctk.CTkOptionMenu(
        bar, values=_LIVE_TASK_OPTIONS, variable=_live_video_task_var,
        font=FONT, height=24, corner_radius=4,
    )
    task_menu.place(relx=0.05, rely=0.56, relwidth=0.10, relheight=0.38)
    Tooltip(
        task_menu,
        "Task type for the YOLO model.\n"
        "'Auto-detect' works for .pt files but will fail for exported\n"
        "formats like TensorRT (.engine) or ONNX (.onnx) that do not\n"
        "embed task metadata.  Select the correct task explicitly\n"
        "(e.g. 'detect' for detection, 'segment' for segmentation).",
    )

    ctk.CTkLabel(bar, text="Seek:", font=FONT, text_color="gray").place(
        relx=0.56, rely=0.56, relwidth=0.04, relheight=0.38
    )

    def _jump(delta_sec: float) -> None:
        fps = _live_video_fps_ref[0]
        cur = _live_video_frame_ref[0]
        total = _live_video_total_ref[0]
        target = max(0, min(total - 1, cur + int(delta_sec * fps)))
        _live_video_seek_to[0] = target

    # Jump buttons: (delta_seconds, label, relative_x_position)
    for delta_seconds, button_label, rel_x_pos in [
        (-10, "−10s", 0.60), (-5, "−5s", 0.65),
        (+5, "+5s",   0.70), (+10, "+10s", 0.76),
    ]:
        ctk.CTkButton(
            bar, text=button_label, command=lambda d=delta_seconds: _jump(d),
            fg_color="#37474f", hover_color="#263238",
            font=("Segoe UI", 10), height=24,
        ).place(relx=rel_x_pos, rely=0.56, relwidth=0.05, relheight=0.38)


def _start_live_video() -> None:
    global _live_video_running, _live_video_cancel_flag, _live_video_paused

    if not _live_video_path:
        messagebox.showerror("Error", "Please select a video file first.")
        return
    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt / .onnx / .engine) first.")
        return

    _live_video_running = True
    _live_video_paused = False
    _live_video_cancel_flag[0] = False
    _live_video_seek_to[0] = -1
    _live_video_frame_ref[0] = 0
    _live_video_total_ref[0] = 1
    _live_video_fps_ref[0] = 25.0

    if _live_video_start_btn:
        _live_video_start_btn.configure(
            text="■  STOP", fg_color="#c62828", hover_color="#b71c1c",
        )
    if _live_video_pause_btn:
        _live_video_pause_btn.configure(text="⏸ Pause", fg_color="#37474f")
    if _live_video_seek_slider:
        _live_video_seek_slider.set(0)

    # ── Audio setup ───────────────────────────────────────────────────────
    audio_want = _live_audio_enabled_var and _live_audio_enabled_var.get()
    is_url = _live_video_is_url[0]
    if audio_want and is_url:
        messagebox.showinfo(
            "Audio Unavailable",
            "Audio playback is not supported for URL/stream sources.\n"
            "It works only with local video files.",
        )
        audio_want = False

    if audio_want:
        # Delegate audio extraction entirely to the video thread's first-frame
        # handler so there is only one code path.  Just show the extracting
        # label here so the user gets immediate visual feedback.
        _safe_label_configure(
            _live_video_status_label,
            text="⏳ Extracting audio…",
            text_color="#64b5f6",
        )

    threading.Thread(target=_live_video_thread, daemon=True).start()


def _stop_live_video() -> None:
    global _live_video_running, _live_video_paused
    _live_video_running = False
    _live_video_paused = False
    _live_video_cancel_flag[0] = True
    _cleanup_live_audio()
    if _live_video_start_btn:
        try:
            _live_video_start_btn.configure(
                text="▶  PLAY", fg_color="#2e7d32", hover_color="#1b5e20",
            )
        except Exception:
            pass
    if _live_video_pause_btn:
        try:
            _live_video_pause_btn.configure(text="⏸ Pause", fg_color="#37474f")
        except Exception:
            pass


def _live_video_thread() -> None:
    """Background thread: open video, run YOLO frame-by-frame, display in label.

    Audio sync strategy
    -------------------
    Raw PCM audio is extracted from the video file once (via ffmpeg) and
    streamed through a sounddevice OutputStream.  The sounddevice callback
    reads ``frames * speed`` source samples per call and resamples them to
    ``frames`` output samples using linear interpolation.  This means speed
    changes take effect within one callback block (~46 ms at 44 100 Hz /
    2 048 frames) without any re-extraction, seek jump, or stutter.

    When the "Sync" checkbox is enabled, the video thread computes a speed EMA
    (alpha = 0.40, converges in ~4 frames) of the ratio
    ``frame_delay / actual_frame_time``.  This is the exact fraction by which
    the video is slower than real-time.  A proportional drift correction
    (``drift_secs / (2 × frame_time)``) is applied on top to close any
    accumulated audio–video position error within roughly two frames.  When Sync
    is re-enabled after being disabled, the audio head is seeked to the current
    video frame position so any drift that built up while sync was off is cleared
    instantly.  ``_pcm_paused`` is managed solely by ``_toggle_pause``; the sync
    path never touches it, so pause/resume works correctly at any point.
    """
    try:
        from ultralytics import YOLO as _YOLO

        device = _get_device()
        raw_task = _live_video_task_var.get() if _live_video_task_var else "Auto-detect"
        task = None if raw_task == "Auto-detect" else raw_task
        model = _YOLO(detection_model_path, task=task)
        _device_controlled = True
        try:
            model.to(device)
        except (RuntimeError, AttributeError, TypeError):
            # TensorRT (.engine) and ONNX models are compiled for a fixed device
            # and cannot be moved.  Skip device= in predict() calls too.
            _device_controlled = False
            import logging
            logging.getLogger(__name__).debug(
                "model.to(device) skipped for %s (compiled model).",
                detection_model_path,
            )

        cap = cv2.VideoCapture(_live_video_path)
        if not cap.isOpened():
            root.after(0, lambda: messagebox.showerror("Error", "Could not open video file."))
            _stop_live_video()
            return

        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        _live_video_total_ref[0] = total_frames
        _live_video_fps_ref[0] = fps

        frame_delay = max(0.001, 1.0 / fps)
        frame_idx = 0
        _audio_was_on = False       # track whether audio was active last iteration
        _audio_sync_was_on = False  # track previous value of audio_sync for edge detection
        _ema_speed = 1.0            # EMA of (frame_delay / actual_frame_time); init real-time

        while not _live_video_cancel_flag[0]:
            # ── Read per-frame controls (take effect immediately) ──────────
            half = _live_video_half_var.get() if _live_video_half_var else False
            conf = _live_video_conf_var.get() if _live_video_conf_var else 0.5
            audio_want = (
                not _live_video_is_url[0]
                and _live_audio_enabled_var is not None
                and _live_audio_enabled_var.get()
            )
            audio_sync = _live_audio_sync_var is not None and _live_audio_sync_var.get()

            # ── Handle audio enabled/disabled while running ────────────────
            stream_active = _pcm_stream is not None
            if audio_want and not _audio_was_on and not stream_active:
                # Audio was just turned on (or was requested at startup).
                # Start extraction in a background thread – this is the ONLY
                # place extraction is triggered so there is no race condition.
                _audio_was_on = True
                vol = _live_audio_volume_var.get() if _live_audio_volume_var else 1.0
                _vid_path = _live_video_path
                root.after(0, lambda: _safe_label_configure(
                    _live_video_status_label,
                    text="⏳ Extracting audio…",
                    text_color="#64b5f6",
                ))

                def _late_stream(video_path=_vid_path, volume=vol):
                    pcm = _pcm_extract(video_path)
                    if pcm is None:
                        root.after(0, lambda: messagebox.showinfo(
                            "Audio Unavailable",
                            "Could not extract audio from the video file.\n\n"
                            "Make sure ffmpeg is installed and on your system PATH:\n"
                            "  Windows: https://ffmpeg.org/download.html\n"
                            "  macOS:   brew install ffmpeg\n"
                            "  Linux:   sudo apt install ffmpeg\n\n"
                            "Video will continue without audio.",
                        ))
                        return
                    start = _live_video_frame_ref[0] / max(_live_video_fps_ref[0], 1.0)
                    ok = _pcm_start_stream(pcm, start_pos_secs=start, volume=volume)
                    if not ok:
                        root.after(0, lambda: messagebox.showinfo(
                            "Audio Unavailable",
                            "Could not open an audio output stream.\n\n"
                            "Please ensure the following packages are installed:\n"
                            "  pip install sounddevice numpy\n\n"
                            "On Linux/macOS you also need PortAudio:\n"
                            "  Linux:  sudo apt install libportaudio2\n"
                            "  macOS:  brew install portaudio\n\n"
                            "Video will continue without audio.",
                        ))

                threading.Thread(target=_late_stream, daemon=True).start()
            elif not audio_want and _audio_was_on:
                _cleanup_live_audio()
                _pcm_speed[0] = 1.0
                _audio_was_on = False
            elif audio_want:
                _audio_was_on = True

            # ── Handle pending seek ────────────────────────────────────────
            seek_target = _live_video_seek_to[0]
            if seek_target >= 0:
                frame_idx = max(0, min(seek_target, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                _live_video_seek_to[0] = -1
                # Seek PCM audio to the corresponding position
                if _pcm_stream is not None:
                    _pcm_seek(frame_idx / fps)

            # ── Pause ─────────────────────────────────────────────────────
            if _live_video_paused:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Copy frames for screenshot use
            _live_video_raw_frame[0] = frame.copy()

            results = model.predict(
                frame, save=False, conf=conf, half=half,
                device=device if _device_controlled else None,
                verbose=False,
            )
            annotated = results[0].plot()
            _live_video_ann_frame[0] = annotated

            # Update shared state
            _live_video_frame_ref[0] = frame_idx
            frac = frame_idx / total_frames

            # ── Measure actual elapsed time for this frame ─────────────────
            t_after_predict = time.perf_counter()
            frame_proc_time = t_after_predict - t_start
            # Total display time = max of inference time and fps throttle delay
            total_frame_time = max(frame_proc_time, frame_delay)
            actual_fps = 1.0 / max(frame_proc_time, 0.001)  # shown in status bar

            # ── Smooth audio speed to match actual video throughput ────────
            # EMA of the speed ratio (alpha=0.4 → converges in ~4 frames).
            # raw_speed is the exact ratio needed for audio to advance at the
            # same rate as the video; < 1.0 when inference is slower than
            # real-time.
            raw_speed = frame_delay / max(total_frame_time, 1e-6)
            _ema_speed = 0.6 * _ema_speed + 0.4 * raw_speed

            if audio_want and _pcm_stream is not None:
                if audio_sync:
                    # If Sync was just re-enabled, seek audio to the current
                    # frame position so any drift accumulated while sync was
                    # off is wiped immediately.
                    if not _audio_sync_was_on:
                        _pcm_seek(frame_idx / fps)

                    # Drift correction: close any remaining positional error
                    # within ~2 frames by adjusting speed proportionally.
                    # Using actual frame time in the denominator makes the
                    # correction self-scaling across fast and slow hardware.
                    expected_sample = int(frame_idx / fps * _PCM_SAMPLE_RATE)
                    drift_secs = (_pcm_pos[0] - expected_sample) / _PCM_SAMPLE_RATE
                    correction = drift_secs / max(2.0 * total_frame_time, 1e-6)

                    _pcm_speed[0] = max(0.05, min(2.0, _ema_speed - correction))
                else:
                    # Sync off: free-run at 1.0×
                    _pcm_speed[0] = 1.0

            _audio_sync_was_on = audio_sync

            # Prepare display image
            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            try:
                lw = max(_live_video_label.winfo_width(),  1)
                lh = max(_live_video_label.winfo_height(), 1)
            except Exception:
                lw, lh = 1280, 720
            scale = min(lw / pil_img.width, lh / pil_img.height)
            nw = max(1, int(pil_img.width  * scale))
            nh = max(1, int(pil_img.height * scale))
            pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)

            det_count = len(results[0].boxes)
            status = (
                f"Frame {frame_idx}/{total_frames}  ({frac * 100:.0f}%)  "
                f"|  {det_count} detection(s)  |  {actual_fps:.1f} fps"
            )

            def _update(ph=photo, st=status, f=frac):
                if _live_video_label:
                    try:
                        _live_video_label.config(image=ph)
                        _live_video_label.image = ph
                    except Exception:
                        pass
                _safe_label_configure(_live_video_status_label, text=st)
                if _live_video_seek_slider and not _live_video_seeking[0]:
                    try:
                        _live_video_seek_slider.set(f)
                    except Exception:
                        pass

            root.after(0, _update)
            frame_idx += 1

            # Throttle to video fps
            elapsed = time.perf_counter() - t_start
            sleep_t = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_t)

        cap.release()
    except Exception as exc:
        _err = str(exc)
        root.after(0, lambda: messagebox.showerror("Live Video Error", _err))
    finally:
        global _live_video_running
        _live_video_running = False
        root.after(0, lambda: _stop_live_video() if _live_video_start_btn else None)


# ─────────────────────────────────────────────────────────────────────────────
#  Export window
# ─────────────────────────────────────────────────────────────────────────────
def _append_export_log(textbox, msg: str) -> None:
    try:
        if textbox and textbox.winfo_exists():
            textbox.insert("end", msg)
            textbox.yview_moveto(1)
    except Exception:
        pass


def show_export_window() -> None:
    global export_model_label, export_status_label, export_model_path
    export_model_path = ""

    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)

    # ── Left: configuration panel ─────────────────────────────────────────
    cfg = ctk.CTkScrollableFrame(
        main_frame,
        label_text="Export Configuration",
        label_font=("Segoe UI", 14, "bold"),
        corner_radius=8,
    )
    cfg.place(relx=0, rely=0, relwidth=0.42, relheight=1.0)

    # ── Right: output log ─────────────────────────────────────────────────
    log_panel = ctk.CTkFrame(main_frame, corner_radius=8)
    log_panel.place(relx=0.43, rely=0, relwidth=0.57, relheight=1.0)

    PAD = {"padx": 14, "pady": 6}

    def _lbl(text):
        l = ctk.CTkLabel(cfg, text=text, font=FLAB, anchor="w")
        l.pack(fill="x", padx=14, pady=(10, 1))
        return l

    def _sep():
        ctk.CTkFrame(cfg, height=1, fg_color="gray50").pack(fill="x", padx=14, pady=4)

    # ── Model selection ────────────────────────────────────────────────────
    _lbl("Trained Model (.pt)")
    sel_btn = ctk.CTkButton(cfg, text="Browse .pt…", font=FBTN, height=38,
                             command=select_export_model)
    sel_btn.pack(fill="x", **PAD)
    Tooltip(sel_btn, "Select the trained YOLO .pt model you want to export.")

    export_model_label = ctk.CTkLabel(
        cfg, text="No model selected", font=("Segoe UI", 11), text_color="gray", anchor="w",
    )
    export_model_label.pack(fill="x", padx=14)
    _sep()

    # ── Export format ──────────────────────────────────────────────────────
    _lbl("Export Format")
    export_fmt_var = ctk.StringVar(value=EXPORT_FORMATS[0])
    fmt_menu = ctk.CTkOptionMenu(
        cfg, variable=export_fmt_var, values=EXPORT_FORMATS,
        font=FBTN, height=38,
    )
    fmt_menu.pack(fill="x", **PAD)
    Tooltip(
        fmt_menu,
        "ONNX            – universal format; runs on CPU, GPU, or accelerators.\n"
        "TensorRT Engine – maximum throughput on NVIDIA GPUs; device-specific.\n"
        "CoreML          – Apple devices (macOS / iOS).\n"
        "TF SavedModel   – TensorFlow ecosystem.\n"
        "TFLite          – mobile / embedded TensorFlow.",
    )
    _sep()

    # ── TensorRT note ──────────────────────────────────────────────────────
    _lbl("ℹ️  Format Notes")
    note_box = ctk.CTkTextbox(
        cfg, font=("Segoe UI", 11), height=160,
        fg_color="#2d2d1e", text_color="#e8e8b0", corner_radius=8,
    )
    note_box.pack(fill="x", **PAD)
    note_box.insert("1.0",
        "ONNX\n"
        "  Universal exchange format.  Runs on CPU, GPU, or hardware\n"
        "  accelerators.  Supported by virtually every inference runtime.\n\n"
        "TensorRT Engine\n"
        "  Compiles the model into GPU-specific machine code for maximum\n"
        "  speed on NVIDIA hardware.  The .engine file is bound to the GPU\n"
        "  it was built on.  Requires TensorRT ≥ 8 (pip install tensorrt).\n\n"
        "CoreML / TFLite\n"
        "  Target Apple or mobile/embedded TensorFlow deployments."
    )
    note_box.configure(state="disabled")
    _sep()

    # ── Export button ──────────────────────────────────────────────────────
    export_status_label = ctk.CTkLabel(
        cfg, text="", font=("Segoe UI", 11), text_color="gray",
        anchor="w", wraplength=260,
    )
    export_status_label.pack(fill="x", padx=14)

    export_btn = ctk.CTkButton(
        cfg,
        text="⬇  Export Model",
        fg_color="#6a1b9a", hover_color="#4a148c",
        font=("Segoe UI", 15, "bold"), height=50,
        text_color="white", corner_radius=8,
    )
    export_btn.pack(fill="x", **PAD)

    # ── Log panel ──────────────────────────────────────────────────────────
    ctk.CTkLabel(
        log_panel, text="Export Output", font=("Segoe UI", 14, "bold"),
    ).pack(anchor="w", padx=12, pady=(10, 4))

    export_log_tb = ctk.CTkTextbox(
        log_panel, font=("Courier New", 11), corner_radius=8,
    )
    export_log_tb.pack(fill="both", expand=True, padx=12, pady=(0, 8))

    export_log_bar = ctk.CTkProgressBar(
        log_panel, progress_color="#6a1b9a", mode="indeterminate",
    )
    export_log_bar.pack(fill="x", padx=12, pady=(0, 10))
    export_log_bar.set(0)

    def _do_export():
        fmt_display = export_fmt_var.get()
        if not export_model_path:
            messagebox.showerror("Error", "Please select a trained .pt model to export.")
            return

        fmt = EXPORT_FORMAT_MAP.get(fmt_display, "onnx")
        export_btn.configure(state="disabled", text="⏳  Exporting…")
        _safe_label_configure(export_status_label, text="Exporting…", text_color="#64b5f6")

        # Clear log
        try:
            export_log_tb.delete("1.0", "end")
        except Exception:
            pass

        export_log_bar.start()
        _append_export_log(export_log_tb, f"Starting export: {Path(export_model_path).name}\n")
        _append_export_log(export_log_tb, f"  Format : {fmt_display}  ({fmt})\n\n")

        script = (
            f"import sys; from ultralytics import YOLO; "
            f"model = YOLO(r'{export_model_path}'); "
            f"out = model.export(format='{fmt}'); "
            f"print('EXPORT_OUTPUT:' + str(out))"
        )
        cmd = [sys.executable, "-c", script]

        def run():
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for raw_line in iter(proc.stdout.readline, ""):
                line = _strip_ansi(raw_line)
                if line.strip():
                    root.after(0, lambda l=line: _append_export_log(export_log_tb, l))
            proc.stdout.close()
            proc.wait()

            if proc.returncode == 0:
                status_msg = "✅  Export finished successfully."
                status_col = "#4caf50"
                btn_text   = "⬇  Export Model"
            else:
                status_msg = f"❌  Export failed (exit code {proc.returncode})."
                status_col = "#ef5350"
                btn_text   = "⬇  Export Model"

            def _done():
                export_log_bar.stop()
                export_log_bar.set(0)
                _append_export_log(export_log_tb, f"\n{status_msg}\n")
                _safe_label_configure(export_status_label, text=status_msg, text_color=status_col)
                try:
                    export_btn.configure(state="normal", text=btn_text)
                except Exception:
                    pass

            root.after(0, _done)

        threading.Thread(target=run, daemon=True).start()

    export_btn.configure(command=_do_export)


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmark window
# ─────────────────────────────────────────────────────────────────────────────

# Column definitions: (header text, pixel width, anchor, tooltip text)
_BENCH_COLS = [
    ("Model",                  175, "w",
     "The model file being benchmarked."),
    ("Accuracy\n(mAP50 ↑)",    100, "center",
     "mAP@50: How accurately the model finds objects when bounding boxes\n"
     "overlap ground truth by ≥50%.\n\nHigher is better. 1.0 = perfect."),
    ("Fine Accuracy\n(mAP50-95 ↑)", 110, "center",
     "mAP@50:95: Stricter accuracy averaged across overlap thresholds\n"
     "from 50% to 95%.  Much harder to score well on.\n\nHigher is better."),
    ("Precision ↑",             95, "center",
     "Precision: Of all detections the model made, what fraction\n"
     "were actually correct?\n\nHigh precision = few false alarms.\nHigher is better."),
    ("Recall ↑",                95, "center",
     "Recall: Of all real objects in the test images, what fraction\n"
     "did the model successfully find?\n\nHigh recall = few missed objects.\nHigher is better."),
    ("Speed\n(ms / img ↓)",     95, "center",
     "Average time in milliseconds to process one image.\n\n"
     "Lower is faster.  Measured on your available hardware (GPU/CPU)."),
    ("Size (MB)",               80, "center",
     "File size of the .pt model on disk (megabytes).\n\n"
     "Smaller models load faster and use less memory."),
]


def show_benchmark_window() -> None:
    global _benchmark_models, _benchmark_results_frame
    global _benchmark_run_btn, _benchmark_model_list_frame

    _benchmark_models = []

    # ── Left: setup panel ─────────────────────────────────────────────────
    setup = ctk.CTkScrollableFrame(
        master=main_frame,
        label_text="Benchmark Setup",
        label_font=("Segoe UI", 14, "bold"),
        corner_radius=8,
    )
    setup.place(relx=0, rely=0, relwidth=0.40, relheight=1.0)

    # ── Right: results panel ──────────────────────────────────────────────
    _benchmark_results_frame = ctk.CTkFrame(master=main_frame, corner_radius=8)
    _benchmark_results_frame.place(relx=0.41, rely=0, relwidth=0.59, relheight=1.0)

    PAD  = {"padx": 14, "pady": 5}
    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)
    FENT = ("Segoe UI", 13)

    def _lbl(text):
        l = ctk.CTkLabel(setup, text=text, font=FLAB, anchor="w")
        l.pack(fill="x", padx=14, pady=(8, 1))
        return l

    def _sep():
        ctk.CTkFrame(setup, height=1, fg_color="gray50").pack(fill="x", padx=14, pady=4)

    # ── Model list ─────────────────────────────────────────────────────────
    _lbl("Models to Benchmark")

    _benchmark_model_list_frame = ctk.CTkScrollableFrame(
        setup, height=130, fg_color="#2a2a3e", corner_radius=6
    )
    _benchmark_model_list_frame.pack(fill="x", padx=14, pady=4)

    def _refresh_model_list():
        for w in _benchmark_model_list_frame.winfo_children():
            w.destroy()
        if not _benchmark_models:
            ctk.CTkLabel(
                _benchmark_model_list_frame,
                text="No models added — click Add Models below.",
                font=("Segoe UI", 11), text_color="gray",
            ).pack(padx=8, pady=8)
            return
        for i, mp in enumerate(_benchmark_models):
            row = ctk.CTkFrame(_benchmark_model_list_frame, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(
                row, text=Path(mp).name, font=("Segoe UI", 11), anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=4)
            ctk.CTkButton(
                row, text="✕", width=28, height=24,
                fg_color="#c62828", hover_color="#b71c1c",
                font=("Segoe UI", 11), text_color="white",
                command=lambda x=i: (_benchmark_models.pop(x), _refresh_model_list()),
            ).pack(side="right", padx=2)

    _refresh_model_list()

    def _add_models():
        paths = filedialog.askopenfilenames(
            title="Select model file(s)",
            filetypes=[
                ("Model files", "*.pt *.onnx *.engine *.trt"),
                ("PyTorch model", "*.pt"),
                ("ONNX model", "*.onnx"),
                ("TensorRT engine", "*.engine *.trt"),
                ("All files", "*.*"),
            ],
        )
        for p in paths:
            p = normalize_path(p)
            if p and p not in _benchmark_models:
                _benchmark_models.append(p)
        _refresh_model_list()

    add_btn = ctk.CTkButton(setup, text="➕  Add Model(s)", font=FBTN, height=36, command=_add_models)
    add_btn.pack(fill="x", **PAD)
    Tooltip(
        add_btn,
        "Add one or more model files to compare.\n"
        "Supported formats: PyTorch (.pt), ONNX (.onnx), TensorRT (.engine / .trt).\n"
        "You can benchmark as many models as you like side-by-side.",
    )
    _sep()

    # ── Dataset source ────────────────────────────────────────────────────
    _lbl("Dataset Source")
    _yaml_ref   = [""]   # mutable container – YAML path
    _folder_ref = [""]   # mutable container – folder path

    yaml_lbl = ctk.CTkLabel(
        setup, text="No YAML selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    folder_lbl = ctk.CTkLabel(
        setup, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )

    # Split radio-button widgets – created further below, referenced here
    split_var   = ctk.StringVar(value="val")
    _split_btns: dict = {}  # keyed by split value string

    def _apply_split_availability(splits: dict, data_source: str) -> None:
        """Enable/disable split radio buttons and auto-select the best one."""
        mapping = {
            "test":   splits.get("test",  False),
            "val":    splits.get("valid", False),
            "train":  splits.get("train", False),
            "folder": data_source == "folder" and (
                splits.get("flat", False) or splits.get("base", False)
            ),
        }
        enabled = []
        for val, ok in mapping.items():
            btn = _split_btns.get(val)
            if btn is None:
                continue
            btn.configure(state="normal" if ok else "disabled")
            if ok:
                enabled.append(val)

        # Auto-select in preference order: test > val > train > folder
        current = split_var.get()
        if current not in enabled:
            for pref in ("test", "val", "train", "folder"):
                if pref in enabled:
                    split_var.set(pref)
                    break

    def _select_yaml():
        p = normalize_path(
            filedialog.askopenfilename(
                title="Select dataset YAML",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            )
        )
        if not p:
            return
        _yaml_ref[0]   = p
        _folder_ref[0] = ""
        yaml_lbl.configure(text=Path(p).name, text_color="#4caf50")
        folder_lbl.configure(text="No folder selected", text_color="gray")
        splits = _detect_yaml_dataset_splits(p)
        _apply_split_availability(splits, "yaml")

    def _select_folder():
        p = normalize_path(filedialog.askdirectory(title="Select dataset folder"))
        if not p:
            return
        splits = _detect_folder_splits(Path(p))
        if not any(splits.values()):
            messagebox.showwarning(
                "No Valid Dataset Found",
                "No valid file or folder structure was found in the selected folder.\n\n"
                "Expected one of:\n"
                "  •  test/images/  +  test/labels/\n"
                "  •  valid/images/ +  valid/labels/\n"
                "  •  train/images/ +  train/labels/\n"
                "  •  images/  +  labels/  (flat structure)\n"
                "  •  Image and matching .txt label files directly in the folder",
            )
            return
        _folder_ref[0] = p
        _yaml_ref[0]   = ""
        folder_lbl.configure(text=Path(p).name, text_color="#4caf50")
        yaml_lbl.configure(text="No YAML selected", text_color="gray")
        _apply_split_availability(splits, "folder")

    yaml_btn = ctk.CTkButton(setup, text="Browse YAML…", font=FBTN, height=36, command=_select_yaml)
    yaml_btn.pack(fill="x", **PAD)
    Tooltip(
        yaml_btn,
        "Select the data.yaml for your dataset.\n\n"
        "This YAML tells YOLO where your images and labels are.\n"
        "Roboflow-exported datasets include data.yaml in the ZIP root.\n\n"
        "After selecting, only the splits that physically exist on disk are enabled.",
    )
    yaml_lbl.pack(fill="x", padx=14)

    folder_btn = ctk.CTkButton(
        setup, text="Browse Folder…", font=FBTN, height=36, command=_select_folder,
    )
    folder_btn.pack(fill="x", **PAD)
    Tooltip(
        folder_btn,
        "Select a folder that contains your dataset images and labels.\n\n"
        "Supported structures (checked in this order):\n"
        "  1.  test/images/   +  test/labels/\n"
        "  2.  valid/images/  +  valid/labels/\n"
        "  3.  train/images/  +  train/labels/\n"
        "  4.  images/        +  labels/  (flat)\n"
        "  5.  Image + .txt files directly in the folder\n\n"
        "The Evaluate On controls below are updated automatically.",
    )
    folder_lbl.pack(fill="x", padx=14)
    _sep()

    # ── Image size ────────────────────────────────────────────────────────
    _lbl("Image Size")
    img_size_entry = ctk.CTkEntry(setup, placeholder_text="640", font=FENT, height=36)
    img_size_entry.pack(fill="x", **PAD)
    Tooltip(
        img_size_entry,
        "Inference resolution in pixels (e.g. 640).\n"
        "Should match the size the model was trained with.",
    )
    _sep()

    # ── Split selector ────────────────────────────────────────────────────
    _lbl("Evaluate On")
    split_frame = ctk.CTkFrame(setup, fg_color="transparent")
    split_frame.pack(fill="x", **PAD)

    _split_defs = [
        ("Test set",    "test",
         "Test set – a held-out split never seen during training.\n"
         "Only available if your dataset has a 'test' folder."),
        ("Valid set",   "val",
         "Valid set – the validation split used while training.\n"
         "Typically always present."),
        ("Train set",   "train",
         "Train set – the training images.\n"
         "Useful to check in-sample performance (expected to be high)."),
        ("Folder set",  "folder",
         "Folder set – use images from the selected folder directly\n"
         "(flat images/ + labels/ structure, or files in the root).\n"
         "Only available when a folder is selected via Browse Folder."),
    ]

    row1 = ctk.CTkFrame(split_frame, fg_color="transparent")
    row1.pack(fill="x")
    row2 = ctk.CTkFrame(split_frame, fg_color="transparent")
    row2.pack(fill="x")

    for idx, (_txt, _val, _tip) in enumerate(_split_defs):
        parent = row1 if idx < 2 else row2
        rb = ctk.CTkRadioButton(
            parent, text=_txt, variable=split_var, value=_val, font=FLAB,
            state="disabled",
        )
        rb.pack(side="left", padx=(0, 16), pady=2)
        Tooltip(rb, _tip)
        _split_btns[_val] = rb

    _sep()

    # ── Run button ────────────────────────────────────────────────────────
    _benchmark_run_btn = ctk.CTkButton(
        setup,
        text="▶  Run Benchmark",
        fg_color="#1565c0", hover_color="#0d47a1",
        font=("Segoe UI", 15, "bold"), height=50,
        text_color="white", corner_radius=8,
        command=lambda: _start_benchmark(img_size_entry, split_var, _yaml_ref, _folder_ref),
    )
    _benchmark_run_btn.pack(fill="x", padx=14, pady=12)

    _show_benchmark_placeholder()


def _show_benchmark_placeholder() -> None:
    global _benchmark_results_frame
    if _benchmark_results_frame is None:
        return
    for w in _benchmark_results_frame.winfo_children():
        w.destroy()
    ctk.CTkLabel(
        _benchmark_results_frame,
        text="📊  Benchmark Results",
        font=("Segoe UI", 20, "bold"),
    ).pack(pady=(40, 10))
    ctk.CTkLabel(
        _benchmark_results_frame,
        text=(
            "Add your trained models and a dataset source on the left,\n"
            "then click  ▶ Run Benchmark  to compare them side-by-side.\n\n"
            "Results will show accuracy, speed, and model size —\n"
            "top performers highlighted in green / blue."
        ),
        font=("Segoe UI", 13),
        text_color="gray",
        justify="center",
    ).pack(pady=4)


def _start_benchmark(img_size_entry, split_var, yaml_ref, folder_ref) -> None:
    global _benchmark_results_frame, _benchmark_run_btn

    yaml_path    = yaml_ref[0]
    folder_path  = folder_ref[0]
    img_size_str = img_size_entry.get().strip() or "640"
    split_ui     = split_var.get()   # 'test' | 'val' | 'train' | 'folder'

    errors = []
    if not _benchmark_models:
        errors.append("• Please add at least one model.")
    if not yaml_path and not folder_path:
        errors.append("• Please select a dataset YAML file or a dataset folder.")
    if not img_size_str.isdigit() or int(img_size_str) < 1:
        errors.append("• Image Size must be a positive integer (e.g. 640).")
    if errors:
        messagebox.showerror("Missing input", "\n".join(errors))
        return

    img_size   = int(img_size_str)
    # Map UI split choice to the string passed to model.val(split=...)
    yolo_split = split_ui if split_ui != "folder" else "val"

    models_run = list(_benchmark_models)
    total_m    = len(models_run)

    if _benchmark_run_btn:
        try:
            _benchmark_run_btn.configure(state="disabled", text="⏳ Running…")
        except Exception:
            pass

    # ── Show log inside results panel ─────────────────────────────────────
    if _benchmark_results_frame is None:
        return
    for w in _benchmark_results_frame.winfo_children():
        w.destroy()

    ctk.CTkLabel(
        _benchmark_results_frame,
        text="⏳  Benchmark in progress…",
        font=("Segoe UI", 15, "bold"),
    ).pack(anchor="w", padx=16, pady=(14, 4))

    bench_progress_lbl = ctk.CTkLabel(
        _benchmark_results_frame, text=f"0 / {total_m} models",
        font=("Segoe UI", 11), text_color="#a6adc8", anchor="w",
    )
    bench_progress_lbl.pack(anchor="w", padx=16)

    log_tb = ctk.CTkTextbox(_benchmark_results_frame, font=("Courier New", 11), corner_radius=8)
    log_tb.pack(fill="both", expand=True, padx=14, pady=(0, 4))

    bench_bar = ctk.CTkProgressBar(
        _benchmark_results_frame, progress_color="#43a047", mode="determinate",
    )
    bench_bar.set(0)
    bench_bar.pack(fill="x", padx=14, pady=(0, 10))

    def _log(msg: str):
        root.after(0, lambda: _bench_append_log(log_tb, msg))

    def _set_progress(done: int):
        frac = done / total_m if total_m > 0 else 1.0
        root.after(0, lambda: bench_bar.set(frac))
        root.after(0, lambda: bench_progress_lbl.configure(text=f"{done} / {total_m} models"))

    def _classify_error(exc: Exception) -> str:
        """Return a user-friendly error message, especially for class-count mismatches."""
        msg = str(exc)
        if "out of bounds for axis" in msg or "index" in msg.lower() and "size" in msg.lower():
            return (
                f"{msg}\n"
                "  ↳ This usually means the model was trained on a different number of classes\n"
                "    than the benchmark dataset contains.  The model cannot be fairly evaluated\n"
                "    on this dataset — results for this model have been skipped."
            )
        return msg

    def run_all():
        import tempfile
        import os as _os
        from ultralytics import YOLO
        all_metrics = []
        _tmp_yamls: list[str] = []   # temp files to clean up after all runs

        for i, mp in enumerate(models_run):
            _log(f"\n[{i + 1}/{total_m}]  Evaluating:  {Path(mp).name}\n")
            _set_progress(i)
            try:
                model = YOLO(mp)

                if folder_path:
                    # ── Folder-based benchmark ──────────────────────────
                    root_p = Path(folder_path)
                    available = _detect_folder_splits(root_p)

                    # Fall back to COCO-like default (80 classes) when the model
                    # doesn't expose a names attribute (e.g. some TRT engines).
                    model_nc    = len(model.names) if hasattr(model, "names") else 80
                    model_names = list(model.names.values()) if hasattr(model, "names") else []

                    effective_yaml = _build_folder_benchmark_yaml(
                        root_p, split_ui, available, model_nc, model_names,
                    )
                    if effective_yaml is None:
                        raise RuntimeError(
                            "Could not determine a valid image path from the selected folder.\n"
                            "Ensure images and labels are present in a supported structure."
                        )
                    _tmp_yamls.append(effective_yaml)
                    _log(f"  Using folder dataset: {folder_path}\n")
                else:
                    # ── YAML-based benchmark ────────────────────────────
                    effective_yaml = yaml_path
                    # Pre-flight: check class count compatibility (PT models only)
                    if mp.lower().endswith(".pt"):
                        yaml_nc   = _get_yaml_nc(effective_yaml)
                        model_nc  = len(model.names) if hasattr(model, "names") else None
                        if yaml_nc is not None and model_nc is not None and model_nc != yaml_nc:
                            raise ValueError(
                                f"Class count mismatch: model has {model_nc} classes but "
                                f"dataset YAML declares {yaml_nc} classes.  "
                                "Benchmark skipped for this model."
                            )

                result = model.val(
                    data=effective_yaml, imgsz=img_size, split=yolo_split, verbose=False,
                )
                m = _extract_bench_metrics(mp, result)
                all_metrics.append(m)
                _log(
                    f"  mAP50={m['map50']:.3f}  mAP50-95={m['map']:.3f}  "
                    f"Speed={m['speed_ms']:.1f} ms/img\n"
                )
            except Exception as exc:
                friendly = _classify_error(exc)
                _log(f"  ❌  Error: {friendly}\n")
                try:
                    sz = Path(mp).stat().st_size / 1_048_576
                except Exception:
                    sz = 0.0
                all_metrics.append({
                    "name": Path(mp).name, "path": mp,
                    "map50": None, "map": None,
                    "precision": None, "recall": None,
                    "speed_ms": None,
                    "size_mb": sz,
                    "error": str(exc),
                })

        _set_progress(total_m)

        # Clean up temporary YAML files generated for folder mode
        for tmp_yaml in _tmp_yamls:
            try:
                _os.unlink(tmp_yaml)
            except Exception:
                pass

        root.after(0, lambda: _finish_benchmark(all_metrics, bench_bar))

    threading.Thread(target=run_all, daemon=True).start()


def _get_yaml_nc(yaml_path: str) -> int | None:
    """Read 'nc' (number of classes) from a YAML file without importing yaml."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("nc:"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return int(parts[1].strip())
    except Exception:
        pass
    return None


_BENCH_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _detect_folder_splits(root: Path) -> dict:
    """Return a dict indicating which dataset splits exist under *root*.

    Keys returned:
      'test'   – ROOT/test/images/ + ROOT/test/labels/ both exist and contain files
      'valid'  – ROOT/valid/images/ + ROOT/valid/labels/ both exist and contain files
      'train'  – ROOT/train/images/ + ROOT/train/labels/ both exist and contain files
      'flat'   – ROOT/images/ + ROOT/labels/ both exist and contain files
      'base'   – image + matching .txt files sit directly inside ROOT/
    """

    def _has_images(d: Path) -> bool:
        try:
            return d.is_dir() and any(
                f.suffix.lower() in _BENCH_IMG_EXTS for f in d.iterdir() if f.is_file()
            )
        except Exception:
            return False

    result: dict = {}

    for split_name in ("test", "valid", "train"):
        img_dir = root / split_name / "images"
        lbl_dir = root / split_name / "labels"
        result[split_name] = _has_images(img_dir) and lbl_dir.is_dir()

    flat_img = root / "images"
    flat_lbl = root / "labels"
    result["flat"] = _has_images(flat_img) and flat_lbl.is_dir()

    try:
        result["base"] = any(
            f.is_file()
            and f.suffix.lower() in _BENCH_IMG_EXTS
            and (root / (f.stem + ".txt")).exists()
            for f in root.iterdir()
        )
    except Exception:
        result["base"] = False

    return result


def _detect_yaml_dataset_splits(yaml_path: str) -> dict:
    """Detect which splits physically exist on disk for a given YAML file.

    Reads the YAML's *path* key (or uses the YAML's parent directory as the
    dataset root) then checks for the standard folder structures.

    Returns the same shape as :func:`_detect_folder_splits` but without
    'flat' and 'base' keys (YAML mode doesn't support those).
    """
    try:
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh) or {}

        yaml_dir = Path(yaml_path).parent
        path_val = data.get("path", "")
        if path_val:
            root = Path(path_val)
            if not root.is_absolute():
                root = (yaml_dir / root).resolve()
        else:
            root = yaml_dir

        # Fall back to yaml_dir when 'path' doesn't point to an existing dir
        if not root.is_dir():
            root = yaml_dir

        splits = _detect_folder_splits(root)
        return {k: splits.get(k, False) for k in ("test", "valid", "train")}
    except Exception:
        return {"test": False, "valid": False, "train": False}


def _build_folder_benchmark_yaml(
    root: Path,
    split_ui: str,
    available_splits: dict,
    nc: int,
    names: list,
) -> str | None:
    """Create a temporary YAML file so Ultralytics can evaluate a plain folder.

    *split_ui* is the value from the split StringVar: 'test', 'val', 'train',
    or 'folder' (meaning the flat/base directory structure).

    Returns the path to the temporary YAML file, or *None* if no valid
    image path could be determined.
    """
    import tempfile
    import yaml as _yaml

    def _abs(p: Path) -> str:
        return str(p).replace("\\", "/")

    data: dict = {
        "path": _abs(root),
        "nc": nc,
        "names": names,
    }

    # Map each YOLO key to its resolved image folder (if available)
    for yolo_key, folder_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
        if available_splits.get(folder_name):
            data[yolo_key] = _abs(root / folder_name / "images")

    # "Folder set" – flat images/ structure or raw base directory
    if split_ui == "folder":
        if available_splits.get("flat"):
            flat_path = _abs(root / "images")
            if "val" not in data:
                data["val"] = flat_path
            if "train" not in data:
                data["train"] = flat_path
        elif available_splits.get("base"):
            base_path = _abs(root)
            if "val" not in data:
                data["val"] = base_path
            if "train" not in data:
                data["train"] = base_path

    # YOLO requires at least a 'val' key
    if "val" not in data:
        for fallback in ("train", "test"):
            if fallback in data:
                data["val"] = data[fallback]
                break
        else:
            return None  # no usable images found

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        prefix="yolo_bench_",
    )
    _yaml.dump(data, tmp, allow_unicode=True, sort_keys=False)
    tmp.close()
    return tmp.name


def _bench_append_log(textbox, msg: str) -> None:
    try:
        if textbox and textbox.winfo_exists():
            textbox.insert("end", msg)
            textbox.yview_moveto(1)
    except Exception:
        pass


def _extract_bench_metrics(model_path: str, result) -> dict:
    """Pull mAP / precision / recall / speed from a val() result object."""
    size_mb = Path(model_path).stat().st_size / 1_048_576

    # Try detection (box) then segmentation (seg)
    box = getattr(result, "box", None) or getattr(result, "seg", None)
    if box is not None:
        map50     = float(getattr(box, "map50", 0) or 0)
        map_val   = float(getattr(box, "map",   0) or 0)
        precision = float(getattr(box, "mp",    0) or 0)
        recall    = float(getattr(box, "mr",    0) or 0)
    else:
        map50 = map_val = precision = recall = 0.0

    speed_dict = getattr(result, "speed", {}) or {}
    speed_ms   = float(speed_dict.get("inference", 0) or 0)

    return {
        "name":      Path(model_path).name,
        "path":      model_path,
        "map50":     map50,
        "map":       map_val,
        "precision": precision,
        "recall":    recall,
        "speed_ms":  speed_ms,
        "size_mb":   size_mb,
    }


def _finish_benchmark(all_metrics: list, bench_bar) -> None:
    global _benchmark_run_btn
    try:
        bench_bar.stop()
    except Exception:
        pass
    if _benchmark_run_btn:
        try:
            _benchmark_run_btn.configure(state="normal", text="▶  Run Benchmark")
        except Exception:
            pass
    try:
        if _benchmark_results_frame and _benchmark_results_frame.winfo_exists():
            _show_benchmark_results(all_metrics)
    except Exception:
        pass


def _show_benchmark_results(metrics_list: list) -> None:
    """Render the results table in the right (results) panel."""
    global _benchmark_results_frame
    if _benchmark_results_frame is None:
        return
    for w in _benchmark_results_frame.winfo_children():
        w.destroy()

    if not metrics_list:
        ctk.CTkLabel(_benchmark_results_frame, text="No results to display.").pack(pady=40)
        return

    # Identify best-in-class models (from successful runs)
    ok = [m for m in metrics_list if m.get("map50") is not None]
    best_acc   = max(ok, key=lambda m: m["map50"])["name"]   if ok else None
    best_speed = min(ok, key=lambda m: m["speed_ms"])["name"] if ok else None
    best_size  = min(ok, key=lambda m: m["size_mb"])["name"]  if ok else None

    # ── Summary banner ─────────────────────────────────────────────────────
    if ok:
        parts = []
        if best_acc:
            parts.append(f"🏆 Most Accurate: {best_acc}")
        if best_speed:
            parts.append(f"⚡ Fastest: {best_speed}")
        if best_size:
            parts.append(f"🪶 Lightest: {best_size}")
        ctk.CTkLabel(
            _benchmark_results_frame,
            text="   |   ".join(parts),
            font=("Segoe UI", 13, "bold"), text_color="#89b4fa",
        ).pack(anchor="w", padx=16, pady=(14, 2))

    # ── Scrollable table ───────────────────────────────────────────────────
    scroll = ctk.CTkScrollableFrame(
        _benchmark_results_frame, corner_radius=8, fg_color="#1e1e2e",
    )
    scroll.pack(fill="both", expand=True, padx=14, pady=(4, 2))

    def _make_row(parent, values_colors, bg):
        row = ctk.CTkFrame(parent, fg_color=bg, corner_radius=4)
        row.pack(fill="x", pady=1)
        for (col_info, (val, color)) in zip(_BENCH_COLS, values_colors):
            _, width, anch, tip = col_info
            lbl = ctk.CTkLabel(
                row, text=val,
                font=("Segoe UI", 12),
                text_color=color or "white",
                justify="center", anchor=anch,
                width=width,
            )
            lbl.pack(side="left", padx=4, pady=6)
            if tip:
                Tooltip(lbl, tip)
        return row

    # Header row
    hdr_vals = [
        (col[0], "#a6adc8") for col in _BENCH_COLS
    ]
    _make_row(scroll, hdr_vals, "#313244")

    # Data rows
    def _fmt(val, fmt="{:.3f}"):
        return "—" if val is None else fmt.format(float(val))

    for idx, m in enumerate(metrics_list):
        is_best_acc   = m["name"] == best_acc
        is_best_speed = m["name"] == best_speed
        is_best_size  = m["name"] == best_size
        bg = "#2a2a3e" if idx % 2 == 0 else "#252535"
        vals_colors = [
            (m["name"],                     "#cdd6f4" if is_best_acc else None),
            (_fmt(m.get("map50")),          "#a6e3a1" if is_best_acc else None),
            (_fmt(m.get("map")),            "#a6e3a1" if is_best_acc else None),
            (_fmt(m.get("precision")),      None),
            (_fmt(m.get("recall")),         None),
            (_fmt(m.get("speed_ms"), "{:.1f}"), "#89dceb" if is_best_speed else None),
            (_fmt(m.get("size_mb"),  "{:.1f}"), "#cba6f7" if is_best_size  else None),
        ]
        _make_row(scroll, vals_colors, bg)

        if m.get("error"):
            err_row = ctk.CTkFrame(scroll, fg_color=bg, corner_radius=0)
            err_row.pack(fill="x")
            ctk.CTkLabel(
                err_row, text=f"  ⚠  {m['error'][:100]}",
                font=("Segoe UI", 10), text_color="#f38ba8", anchor="w",
            ).pack(anchor="w", padx=8, pady=(0, 4))

    # ── Legend ─────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        _benchmark_results_frame,
        text=(
            "🟢 Green = most accurate   🔵 Blue = fastest   🟣 Purple = smallest   "
            "↑ higher is better   ↓ lower is better"
        ),
        font=("Segoe UI", 10), text_color="#6c7086",
    ).pack(anchor="w", padx=16, pady=(2, 4))

    # ── Chart button ────────────────────────────────────────────────────────
    ok_metrics = [m for m in metrics_list if m.get("map50") is not None]
    if ok_metrics:
        ctk.CTkButton(
            _benchmark_results_frame,
            text="📊  Show Bar Charts",
            fg_color="#1565c0", hover_color="#0d47a1",
            font=("Segoe UI", 12, "bold"), height=34,
            command=lambda: _show_benchmark_chart(metrics_list),
        ).pack(anchor="w", padx=16, pady=(0, 10))


def _show_benchmark_chart(metrics_list: list) -> None:
    """Open a Toplevel window with bar charts for accuracy, speed, and model size."""
    ok = [m for m in metrics_list if m.get("map50") is not None]
    if not ok:
        messagebox.showinfo("Chart", "No successful benchmark results to chart.")
        return

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    except ImportError:
        messagebox.showinfo(
            "matplotlib not found",
            "Install matplotlib to view charts:\n\n  pip install matplotlib\n\n"
            "Then restart the application.",
        )
        return

    win = tk.Toplevel(root)
    win.title("Benchmark Charts")
    win.geometry("960x540")
    win.configure(bg="#1e1e2e")

    names     = [m["name"][:18] for m in ok]
    map50     = [m["map50"]    for m in ok]
    speed     = [m["speed_ms"] for m in ok]
    size_mb   = [m["size_mb"]  for m in ok]
    map5095   = [m["map"]      for m in ok]

    BG   = "#1e1e2e"
    AXIS = "#2a2a3e"
    TXT  = "white"
    xs   = range(len(names))

    fig = Figure(figsize=(9.6, 5.0), dpi=100, facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.26, wspace=0.35)

    axes_defs = [
        (221, map50,   "#a6e3a1", "Accuracy (mAP50 ↑)"),
        (222, map5095, "#89b4fa", "Fine Accuracy (mAP50-95 ↑)"),
        (223, speed,   "#89dceb", "Speed (ms / img ↓)"),
        (224, size_mb, "#cba6f7", "Model Size (MB)"),
    ]

    for pos, data, colour, title in axes_defs:
        ax = fig.add_subplot(pos)
        ax.set_facecolor(AXIS)
        ax.tick_params(colors=TXT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#45475a")
        bars = ax.bar(xs, data, color=colour, zorder=2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7, color=TXT)
        ax.set_title(title, color=TXT, fontsize=9, pad=4)
        ax.yaxis.label.set_color(TXT)
        ax.title.set_color(TXT)
        # Value labels on top of each bar
        for bar, val in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=6, color=TXT,
            )

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    tb_frame = ctk.CTkFrame(win, fg_color="#1e1e2e", height=32)
    tb_frame.pack(fill="x")
    tb = NavigationToolbar2Tk(canvas, tb_frame)
    tb.update()


# ─────────────────────────────────────────────────────────────────────────────
#  File / folder selection dialogs
# ─────────────────────────────────────────────────────────────────────────────
def select_train_data() -> None:
    global train_data_path, train_data_label
    initial = _browse_last_dirs.get("train_data", "")
    path = normalize_path(filedialog.askdirectory(
        title="Select Training Data Folder",
        initialdir=initial if initial else None,
    ))
    if path:
        train_data_path = path
        _browse_last_dirs["train_data"] = str(Path(path).parent)
        short = Path(path).name or path
        _safe_label_configure(train_data_label, text=short, text_color="#4caf50")
        if _train_data_btn_ref[0]:
            try:
                _train_data_btn_ref[0].configure(fg_color="#2e7d32", hover_color="#1b5e20")
            except Exception:
                pass
        _auto_load_training_yaml(path)


def select_model_save_folder() -> None:
    global model_save_path, model_save_label
    initial = _browse_last_dirs.get("model_save", "")
    path = normalize_path(filedialog.askdirectory(
        title="Select Model Save Folder",
        initialdir=initial if initial else None,
    ))
    if path:
        model_save_path = path
        _browse_last_dirs["model_save"] = str(Path(path).parent)
        short = Path(path).name or path
        _safe_label_configure(model_save_label, text=short, text_color="#4caf50")
        if _model_save_btn_ref[0]:
            try:
                _model_save_btn_ref[0].configure(fg_color="#2e7d32", hover_color="#1b5e20")
            except Exception:
                pass


def select_custom_model() -> None:
    global custom_model_path, custom_model_label
    initial = _browse_last_dirs.get("custom_model", "")
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select Custom Base Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=initial if initial else None,
        )
    )
    if path:
        custom_model_path = path
        _browse_last_dirs["custom_model"] = str(Path(path).parent)
        _safe_label_configure(
            custom_model_label,
            text=f"Custom: {Path(path).name}",
            text_color="#64b5f6",
        )
        if _custom_model_btn_ref[0]:
            try:
                _custom_model_btn_ref[0].configure(fg_color="#2e7d32", hover_color="#1b5e20")
            except Exception:
                pass


def clear_custom_model() -> None:
    global custom_model_path, custom_model_label
    custom_model_path = ""
    _safe_label_configure(
        custom_model_label,
        text="Using built-in pretrained weights",
        text_color="gray",
    )


def select_detection_images_folder() -> None:
    global detection_images_folder_path, detect_folder_label, image_paths, current_image_index
    global _detect_progress_value, _detect_progress_text, _detect_file_count_text
    path = normalize_path(filedialog.askdirectory(title="Select Images/Videos Folder"))
    if not path:
        return

    detection_images_folder_path = path
    short = Path(path).name or path
    _safe_label_configure(detect_folder_label, text=short, text_color="#4caf50")

    # Clear any previous results
    image_paths = []
    current_image_index = 0
    if image_label:
        try:
            image_label.config(image="")
            image_label.image = None
        except Exception:
            pass
    _safe_label_configure(image_index_label, text="No results")
    if detection_progress_bar:
        try:
            detection_progress_bar.set(0)
        except Exception:
            pass
    _safe_label_configure(_detect_progress_label, text="")
    _detect_progress_value = 0.0
    _detect_progress_text  = ""

    # Count files in background (can be slow for large folders)
    def _count():
        global _detect_file_count_text
        try:
            imgs, vids = get_media_files(path)
            txt = f"{len(imgs)} image(s),  {len(vids)} video(s)"
        except Exception:
            txt = "Could not count files"
        _detect_file_count_text = txt
        root.after(0, lambda: _safe_label_configure(_detect_file_count_label, text=txt))

    threading.Thread(target=_count, daemon=True).start()
    _detect_file_count_text = "Counting files…"
    _safe_label_configure(_detect_file_count_label, text="Counting files…")


def select_detection_model() -> None:
    global detection_model_path, detect_model_label, image_paths, current_image_index
    global _detect_progress_value, _detect_progress_text, _detect_model_info_text
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[
                ("YOLO model", "*.pt *.engine *.onnx *.tflite *.pb"),
                ("PyTorch weights", "*.pt"),
                ("TensorRT engine", "*.engine"),
                ("ONNX model", "*.onnx"),
                ("All files", "*.*"),
            ],
        )
    )
    if not path:
        return

    detection_model_path = path
    _safe_label_configure(detect_model_label, text=Path(path).name, text_color="#4caf50")

    # Clear any previous results
    image_paths = []
    current_image_index = 0
    if image_label:
        try:
            image_label.config(image="")
            image_label.image = None
        except Exception:
            pass
    _safe_label_configure(image_index_label, text="No results")
    if detection_progress_bar:
        try:
            detection_progress_bar.set(0)
        except Exception:
            pass
    _safe_label_configure(_detect_progress_label, text="")
    _detect_progress_value = 0.0
    _detect_progress_text  = ""
    _detect_model_info_text = "Loading model info…"
    _safe_label_configure(_detect_model_info_label, text="Loading model info…", text_color="#6c7086")

    # Load model info in background thread
    def _load_info():
        try:
            info = get_model_info(path)
            nc   = info["num_classes"]
            task = info["task"]
            sz   = info["size_mb"]
            if nc is not None:
                classes_preview = ", ".join(info["class_names"][:5])
                if len(info["class_names"]) > 5:
                    classes_preview += f" … (+{len(info['class_names']) - 5} more)"
                txt = (
                    f"Size: {sz:.1f} MB  |  Task: {task}\n"
                    f"{nc} class(es): {classes_preview}"
                )
            else:
                txt = f"Size: {sz:.1f} MB"
            # Auto-populate the task dropdown when the model reports a known task
            _KNOWN_TASKS = {"detect", "segment", "classify", "pose", "obb"}
            auto_task = task if task in _KNOWN_TASKS else None
        except Exception as exc:
            txt = f"Could not load info: {exc}"
            auto_task = None

        def _apply(t=txt, at=auto_task):
            global _detect_model_info_text
            _detect_model_info_text = t
            _safe_label_configure(_detect_model_info_label, text=t, text_color="#a6adc8")
            if at is not None and _detect_task_var is not None:
                _detect_task_var.set(at)

        root.after(0, _apply)

    threading.Thread(target=_load_info, daemon=True).start()


def select_camera_save_folder() -> None:
    global detection_save_dir
    path = normalize_path(filedialog.askdirectory(title="Select Capture Save Folder"))
    if path:
        detection_save_dir = path
        if camera_detection:
            camera_detection.set_save_directory(path)


def select_export_model() -> None:
    global export_model_path, export_model_label
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
    )
    if path:
        export_model_path = path
        _safe_label_configure(
            export_model_label, text=Path(path).name, text_color="#4caf50"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Training logic
# ─────────────────────────────────────────────────────────────────────────────
def enqueue_output(out, queue: Queue) -> None:
    for line in iter(out.readline, ""):
        queue.put(line)
    out.close()


def start_training(
    project_name_entry,
    input_size_entry,
    epochs_entry,
    batch_size_entry,
    class_names_text,
    workers_entry=None,
) -> None:
    global project_name, train_data_path, model_save_path, custom_model_path
    global input_size, epochs, batch_size, class_names

    project_name = project_name_entry.get().strip()
    input_size   = input_size_entry.get().strip()
    epochs_val   = epochs_entry.get().strip()
    batch_val    = batch_size_entry.get().strip()
    raw_classes  = class_names_text.get("1.0", "end-1c")
    class_names  = [n.strip() for n in raw_classes.splitlines() if n.strip()]
    workers_val  = workers_entry.get().strip() if workers_entry else "8"

    selected_display    = selected_model_var.get() if selected_model_var else ""
    selected_model_size = MODEL_MAP.get(selected_display, "")

    # Validate inputs
    errors = []
    if not project_name:
        errors.append("• Project Name is empty.")
    if not model_save_path:
        errors.append("• Model Save Folder not selected.")
    if not selected_model_size and not custom_model_path:
        errors.append("• No YOLO model selected and no custom model loaded.")
    if not input_size or not input_size.isdigit() or int(input_size) < 1:
        errors.append("• Image Size must be a positive integer (e.g. 640).")
    if not epochs_val or not epochs_val.isdigit() or int(epochs_val) < 1:
        errors.append("• Epochs must be a positive integer (e.g. 300).")
    if not batch_val or not batch_val.isdigit() or int(batch_val) < 1:
        errors.append("• Batch Size must be a positive integer (e.g. 16).")
    if not class_names:
        errors.append("• Class Names are empty.")

    # When using a Roboflow YAML we don't need a raw data folder
    if not roboflow_yaml_path and not train_data_path:
        errors.append("• Training Data Folder not selected (or import a Roboflow ZIP).")

    if errors:
        messagebox.showerror("Missing / invalid input", "\n".join(errors))
        return

    epochs     = epochs_val
    batch_size = batch_val

    # Collect extra params from advanced options widgets
    try:
        extra_params = _collect_extra_params_global(workers_val)
    except Exception:
        extra_params = {'workers': int(workers_val) if workers_val.isdigit() else 8}

    # ── Check for existing training checkpoint ─────────────────────────────
    _check_and_offer_resume(project_name, epochs_val, extra_params)
    return  # _check_and_offer_resume calls _proceed_with_training internally


def _collect_extra_params_global(workers_val="8"):
    """Collect advanced training params from module-level vars."""
    ep = {}
    ep['workers'] = int(workers_val) if str(workers_val).isdigit() else 8
    try:
        ep['time'] = int(_train_time_var.get()) if _train_time_var else 0
    except Exception:
        ep['time'] = 0
    try:
        ep['patience'] = int(_train_patience_var.get()) if _train_patience_var else 100
    except Exception:
        ep['patience'] = 100
    ep['save'] = bool(_train_save_var.get()) if _train_save_var else True
    try:
        ep['save_period'] = int(_train_save_period_var.get()) if _train_save_period_var else -1
    except Exception:
        ep['save_period'] = -1
    ep['cache'] = bool(_train_cache_var.get()) if _train_cache_var else False
    ep['resume'] = bool(_train_resume_var.get()) if _train_resume_var else False
    try:
        ep['freeze'] = int(_train_freeze_var.get()) if _train_freeze_var else 0
    except Exception:
        ep['freeze'] = 0
    try:
        ep['lr0'] = float(_train_lr0_var.get()) if _train_lr0_var else 0.01
    except Exception:
        ep['lr0'] = 0.01
    try:
        ep['lrf'] = float(_train_lrf_var.get()) if _train_lrf_var else 0.01
    except Exception:
        ep['lrf'] = 0.01
    try:
        ep['momentum'] = float(_train_momentum_var.get()) if _train_momentum_var else 0.937
    except Exception:
        ep['momentum'] = 0.937
    try:
        ep['weight_decay'] = float(_train_weight_decay_var.get()) if _train_weight_decay_var else 0.0005
    except Exception:
        ep['weight_decay'] = 0.0005
    ep['optimizer'] = str(_train_optimizer_var.get()) if _train_optimizer_var else 'auto'
    ep['val'] = bool(_train_val_var.get()) if _train_val_var else True
    try:
        ep['max_det'] = int(_train_max_det_var.get()) if _train_max_det_var else 300
    except Exception:
        ep['max_det'] = 300
    return ep


def _check_and_offer_resume(project_name: str, epochs_val: str, extra_params: dict) -> None:
    """Check for existing checkpoint and offer resume, then proceed."""
    runs_base = Path("runs")
    last_pt = None
    checkpoint_dir = None
    for task_dir in ["detect", "segment", "classify", "pose", "obb", "train"]:
        candidate = runs_base / task_dir / project_name / "weights" / "last.pt"
        if candidate.exists():
            last_pt = candidate
            checkpoint_dir = candidate.parent.parent
            break

    if last_pt is None:
        _proceed_with_training(extra_params)
        return

    epochs_done_msg = ""
    try:
        import torch as _torch
        ckpt = _torch.load(str(last_pt), map_location="cpu", weights_only=False)
        epoch_done = ckpt.get("epoch", None)
        if epoch_done is not None:
            remaining = int(epochs_val) - int(epoch_done) - 1
            epochs_done_msg = (
                f"\n\nCheckpoint epoch: {epoch_done + 1}\n"
                f"Target epochs: {epochs_val}\n"
                f"Remaining epochs: {max(0, remaining)}"
            )
    except Exception:
        pass

    answer = messagebox.askyesnocancel(
        "Resume Training?",
        f"A previous training checkpoint was found for project '{project_name}':\n"
        f"{checkpoint_dir}{epochs_done_msg}\n\n"
        "• YES – Resume training from the last checkpoint\n"
        "• NO – Start fresh (ignores checkpoint)\n"
        "• CANCEL – Abort",
    )

    if answer is None:
        return
    if answer:
        try:
            _TEMP_DIR.mkdir(exist_ok=True)
            temp_last = _TEMP_DIR / f"resume_{project_name}_last.pt"
            import shutil as _shutil
            _shutil.copy2(str(last_pt), str(temp_last))
            extra_params = dict(extra_params)
            extra_params['resume'] = True
            output_queue.put(f"⏩ Resuming training from checkpoint: {last_pt}\n")
            output_queue.put(f"   Temp copy created at: {temp_last}\n")
        except Exception as exc:
            messagebox.showerror("Resume Error", f"Could not copy checkpoint:\n{exc}")
            return
    _proceed_with_training(extra_params)


def _proceed_with_training(extra_params: dict) -> None:
    """Build YAML and kick off the training subprocess."""
    workers_int = extra_params.get('workers', 8)

    if roboflow_yaml_path:
        yaml_path = roboflow_yaml_path
    else:
        yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)

    selected_display    = selected_model_var.get() if selected_model_var else ""
    selected_model_size = MODEL_MAP.get(selected_display, "")

    _run_training_subprocess(yaml_path, selected_model_size, workers_int, extra_params)


def _stop_training() -> None:
    """Terminate the running training subprocess."""
    proc = _train_proc[0]
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            output_queue.put("\n⏹ Training stop requested. Waiting for current epoch to finish…\n")
        except Exception as e:
            output_queue.put(f"\n⚠ Could not stop training: {e}\n")
    else:
        output_queue.put("\n⚠ No training process is currently running.\n")


def _run_training_subprocess(
    yaml_path: str, selected_model_size: str, workers_int: int = 8,
    extra_params: dict = None,
) -> None:
    global progress_bar, output_textbox, _train_log_buffer, _train_progress_value, _train_progress_text

    # Clear the log buffer so previous training runs don't accumulate indefinitely
    _train_log_buffer.clear()
    _train_progress_value = 0.0
    _train_progress_text  = ""

    import json as _json
    extra_json = _json.dumps(extra_params or {})

    cmd = [
        sys.executable, "src/train.py",
        project_name,
        train_data_path,
        ",".join(class_names),
        model_save_path,
        selected_model_size,
        str(input_size),
        str(epochs),
        yaml_path,
        str(batch_size),
        custom_model_path,
        extra_json,
    ]

    # Pattern: lines like "      1/100  " at the start
    epoch_re = re.compile(r'^\s*(\d+)/(\d+)\s')

    def _update_train_progress(current_ep: int, total_ep: int) -> None:
        global _train_progress_value, _train_progress_text
        if progress_bar is None:
            return
        try:
            if not progress_bar.winfo_exists():
                return
            frac = current_ep / max(total_ep, 1)
            _train_progress_value = frac
            _train_progress_text = f"Epoch {current_ep} / {total_ep}  ({frac * 100:.0f}%)"
            progress_bar.set(frac)
            lbl = getattr(progress_bar, "_progress_label", None)
            if lbl:
                try:
                    lbl.configure(text=_train_progress_text)
                except Exception:
                    pass
        except Exception:
            pass

    def _set_stop_btn(enabled: bool) -> None:
        btn = _train_stop_btn_ref[0]
        if btn:
            try:
                if btn.winfo_exists():
                    btn.configure(state="normal" if enabled else "disabled")
            except Exception:
                pass

    def run() -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _train_proc[0] = proc
        root.after(0, lambda: _set_stop_btn(True))

        def _reader():
            for raw_line in iter(proc.stdout.readline, ""):
                line = _strip_ansi(raw_line)
                line = line.lstrip()
                # Skip lines that were purely escape sequences
                if line.strip():
                    output_queue.put(line)
                m = epoch_re.match(raw_line)
                if not m:
                    m = epoch_re.match(line)
                if m:
                    ep_cur = int(m.group(1))
                    ep_tot = int(m.group(2))
                    root.after(0, lambda c=ep_cur, t=ep_tot: _update_train_progress(c, t))
            proc.stdout.close()

        threading.Thread(target=_reader, daemon=True).start()
        proc.wait()
        _train_proc[0] = None
        root.after(0, lambda: _set_stop_btn(False))
        root.after(0, _training_finished)

    if progress_bar:
        try:
            progress_bar.set(0)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def _training_finished() -> None:
    global progress_bar, output_textbox, _train_progress_value, _train_progress_text
    _train_progress_value = 1.0
    _train_progress_text = "Training complete ✅"
    if progress_bar:
        try:
            progress_bar.set(1.0)
            lbl = getattr(progress_bar, "_progress_label", None)
            if lbl:
                lbl.configure(text=_train_progress_text)
        except Exception:
            pass
    done_msg = "\n✅ Training process finished.\n"
    _train_log_buffer.append(done_msg)
    try:
        if output_textbox and output_textbox.winfo_exists():
            output_textbox.insert("end", done_msg)
            output_textbox.yview_moveto(1)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Image / Video detection logic
# ─────────────────────────────────────────────────────────────────────────────
def toggle_image_detection() -> None:
    """Toggle between Start Detection and Cancel Detection."""
    global _detection_running
    if _detection_running:
        _cancel_image_detection()
    else:
        _begin_image_detection()


def _begin_image_detection() -> None:
    global _detection_running, _detection_cancel_flag

    if not detection_images_folder_path:
        messagebox.showerror("Error", "Please select an images/videos folder first.")
        return
    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt file) first.")
        return

    _detection_running = True
    _detection_cancel_flag[0] = False

    # Update button appearance
    if _detect_start_btn:
        _detect_start_btn.configure(
            text="⛔  Cancel Detection",
            fg_color="#c62828",
            hover_color="#b71c1c",
        )

    # Disable other controls
    for w in _detect_controls:
        try:
            w.configure(state="disabled")
        except Exception:
            pass

    # Reset progress bar
    if detection_progress_bar:
        try:
            detection_progress_bar.set(0)
        except Exception:
            pass
    _safe_label_configure(_detect_progress_label, text="Starting…")

    # Collect options
    conf = _detect_conf_var.get() if _detect_conf_var else 0.5
    half = _detect_half_var.get() if _detect_half_var else False
    try:
        workers_val = int(_detect_workers_var.get()) if _detect_workers_var else 4
    except Exception:
        workers_val = 4
    raw_task = _detect_task_var.get() if _detect_task_var else "Auto-detect"
    task_val = None if raw_task == "Auto-detect" else raw_task

    def _progress_cb(current: int, total: int, msg: str) -> None:
        frac = current / max(total, 1)
        root.after(0, lambda: _update_detect_progress(frac, current, total, msg))

    def _image_result_cb(result_path: str) -> None:
        """Called by detect_images after each image is saved – display it immediately."""
        root.after(0, lambda p=result_path: _show_single_result(p))

    def _run_detect():
        """Thread target: run detection and guarantee the UI is always reset."""
        _completed = [False]

        def _wrapped_callback(results_dir: str) -> None:
            _completed[0] = True
            _on_detection_complete(results_dir)

        try:
            detect_images(
                images_folder=detection_images_folder_path,
                model_path=detection_model_path,
                callback=_wrapped_callback,
                progress_callback=_progress_cb,
                image_result_callback=_image_result_cb,
                conf_threshold=conf,
                half=half,
                workers=workers_val,
                cancel_flag=lambda: _detection_cancel_flag[0],
                task=task_val,
                device=_get_device(),
            )
        except Exception as exc:
            root.after(0, lambda: messagebox.showerror("Detection Error", str(exc)))
        finally:
            # If the callback was never called (e.g. error or empty folder), reset UI
            if not _completed[0]:
                root.after(0, _show_detection_results)

    threading.Thread(target=_run_detect, daemon=True).start()


def _cancel_image_detection() -> None:
    global _detection_cancel_flag
    _detection_cancel_flag[0] = True
    _safe_label_configure(_detect_progress_label, text="Cancelling…")


def _update_detect_progress(frac: float, current: int, total: int, msg: str) -> None:
    global _detect_progress_value, _detect_progress_text
    _detect_progress_value = frac
    _detect_progress_text = f"{frac * 100:.0f}%  ({current}/{total})  {msg}"
    if detection_progress_bar:
        try:
            detection_progress_bar.set(frac)
        except Exception:
            pass
    _safe_label_configure(_detect_progress_label, text=_detect_progress_text)
    _safe_label_configure(image_index_label, text=f"{current}/{total}")


def _show_single_result(result_path: str) -> None:
    """Add a newly processed result image to image_paths and display it immediately."""
    global image_paths, current_image_index
    if result_path not in image_paths:
        image_paths.append(result_path)
        # Jump to the latest image so the user sees each result as it arrives
        current_image_index = len(image_paths) - 1
    n = len(image_paths)
    _safe_label_configure(image_index_label, text=f"{current_image_index + 1}/{n}")
    update_image()


def _on_detection_complete(results_dir: str) -> None:
    global image_paths, current_image_index
    # Rebuild the full sorted list (catches any images the per-image callback may
    # have missed, e.g. for video frame results) and stay on the last viewed image.
    full_list = sorted(
        str(p) for p in Path(results_dir).iterdir()
        if p.is_file() and is_valid_image(str(p))
    )
    if full_list:
        # Keep the user's current position if images were already displayed
        prev_index = current_image_index
        image_paths = full_list
        current_image_index = min(prev_index, len(image_paths) - 1)
    else:
        image_paths = []
        current_image_index = 0
    root.after(0, _show_detection_results)


def _show_detection_results() -> None:
    global _detection_running, _detect_progress_value, _detect_progress_text
    _detection_running = False

    # Restore start button
    if _detect_start_btn:
        try:
            _detect_start_btn.configure(
                text="▶  Start Detection",
                fg_color="#1565c0",
                hover_color="#0d47a1",
            )
        except Exception:
            pass

    # Re-enable controls
    for w in _detect_controls:
        try:
            w.configure(state="normal")
        except Exception:
            pass

    _detect_progress_value = 1.0 if image_paths else 0.0
    if detection_progress_bar:
        try:
            detection_progress_bar.set(_detect_progress_value)
        except Exception:
            pass

    if image_paths:
        n = len(image_paths)
        _detect_progress_text = f"Done – {n} result image(s)"
        _safe_label_configure(_detect_progress_label, text=_detect_progress_text)
        _safe_label_configure(image_index_label, text=f"{current_image_index + 1}/{n}")
        update_image()
    else:
        _detect_progress_text = "No result images found"
        _safe_label_configure(_detect_progress_label, text=_detect_progress_text)
        _safe_label_configure(image_index_label, text="No results")


# ─────────────────────────────────────────────────────────────────────────────
#  Gallery and fullscreen image viewers
# ─────────────────────────────────────────────────────────────────────────────
def _open_gallery() -> None:
    """Open a Toplevel window showing all result images in a thumbnail grid."""
    if not image_paths:
        messagebox.showinfo("Gallery", "Run detection first to generate result images.")
        return

    win = tk.Toplevel(root)
    win.title(f"Detection Gallery  ({len(image_paths)} images)")
    win.geometry("1100x700")
    win.configure(bg="#1e1e2e")

    THUMB_SIZE = 180
    COLS = 5

    header = ctk.CTkLabel(
        win,
        text=f"🖼  Detection Gallery  —  {len(image_paths)} result image(s)",
        font=("Segoe UI", 14, "bold"),
    )
    header.pack(pady=(10, 4))

    scroll_frame = ctk.CTkScrollableFrame(win, corner_radius=0, fg_color="#1e1e2e")
    scroll_frame.pack(fill="both", expand=True, padx=8, pady=8)

    row_frame = None
    _thumb_refs = []  # keep PhotoImage refs alive

    def _open_full(idx: int) -> None:
        _open_fullscreen_image(start_index=idx)

    for i, path in enumerate(image_paths):
        col = i % COLS
        if col == 0:
            row_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=4)

        cell = ctk.CTkFrame(row_frame, fg_color="#2a2a3e", corner_radius=6)
        cell.pack(side="left", padx=6)

        try:
            img = Image.open(path)
            img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
        except Exception:
            photo = None

        if photo:
            _thumb_refs.append(photo)
            btn = tk.Button(
                cell, image=photo, bg="#2a2a3e", relief="flat",
                cursor="hand2",
                command=lambda idx=i: _open_full(idx),
            )
            btn.image = photo
            btn.pack(padx=4, pady=(4, 2))
        else:
            ctk.CTkLabel(cell, text="⚠ load error", font=("Segoe UI", 10)).pack(
                padx=4, pady=(4, 2)
            )

        name_lbl = ctk.CTkLabel(
            cell, text=Path(path).name[:24], font=("Segoe UI", 9), text_color="#a6adc8"
        )
        name_lbl.pack(padx=4, pady=(0, 4))

    # store refs on window to prevent GC
    win._thumb_refs = _thumb_refs


def _open_fullscreen_image(start_index: int | None = None) -> None:
    """Open a fullscreen image viewer with prev/next navigation."""
    if not image_paths:
        return

    idx = [start_index if start_index is not None else current_image_index]
    idx[0] = max(0, min(idx[0], len(image_paths) - 1))

    win = tk.Toplevel(root)
    win.title("Fullscreen Detection Viewer")
    sw, sh = get_screen_size()
    win.geometry(f"{sw}x{sh}")
    win.configure(bg="#000000")
    win.attributes("-fullscreen", False)

    img_lbl = tk.Label(win, bg="#000000")
    img_lbl.place(relx=0, rely=0, relwidth=1.0, relheight=0.94)

    nav = ctk.CTkFrame(win, corner_radius=0, fg_color="#1e1e2e", height=44)
    nav.place(relx=0, rely=0.94, relwidth=1.0, relheight=0.06)

    idx_lbl = ctk.CTkLabel(nav, text="", font=("Segoe UI", 13))
    idx_lbl.place(relx=0.44, rely=0.1, relwidth=0.12, relheight=0.8)

    def _show(i: int) -> None:
        idx[0] = i % len(image_paths)
        try:
            pil_img = Image.open(image_paths[idx[0]])
            dw = max(win.winfo_width(),  1)
            dh = max(int(win.winfo_height() * 0.94), 1)
            scale = min(dw / pil_img.width, dh / pil_img.height)
            nw = max(1, int(pil_img.width  * scale))
            nh = max(1, int(pil_img.height * scale))
            pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_img)
            img_lbl.config(image=photo)
            img_lbl.image = photo
        except Exception:
            pass
        idx_lbl.configure(text=f"{idx[0] + 1} / {len(image_paths)}")

    ctk.CTkButton(
        nav, text="◀ Prev", command=lambda: _show(idx[0] - 1),
        fg_color="#1976d2", font=("Segoe UI", 13, "bold"), height=34,
    ).place(relx=0.36, rely=0.1, relwidth=0.08, relheight=0.8)

    ctk.CTkButton(
        nav, text="Next ▶", command=lambda: _show(idx[0] + 1),
        fg_color="#1976d2", font=("Segoe UI", 13, "bold"), height=34,
    ).place(relx=0.56, rely=0.1, relwidth=0.08, relheight=0.8)

    ctk.CTkButton(
        nav, text="✕  Close", command=win.destroy,
        fg_color="#c62828", hover_color="#b71c1c",
        font=("Segoe UI", 13, "bold"), height=34,
    ).place(relx=0.85, rely=0.1, relwidth=0.10, relheight=0.8)

    win.bind("<Left>",  lambda _e: _show(idx[0] - 1))
    win.bind("<Right>", lambda _e: _show(idx[0] + 1))
    win.bind("<Escape>", lambda _e: win.destroy())

    win.after(100, lambda: _show(idx[0]))


# ─────────────────────────────────────────────────────────────────────────────
#  Training queue runner
# ─────────────────────────────────────────────────────────────────────────────
def _run_training_queue() -> None:
    """Run all queued training jobs sequentially in a background thread."""
    global _train_queue_running

    if _train_queue_running:
        messagebox.showinfo("Queue", "A queue run is already in progress.")
        return
    if not _train_queue:
        messagebox.showinfo("Queue", "The training queue is empty.")
        return

    jobs = list(_train_queue)
    _train_queue.clear()
    if _train_queue_frame:
        root.after(0, lambda: _refresh_queue_list_safe())

    _train_queue_running = True

    def _refresh_queue_list_safe():
        for w in _train_queue_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            _train_queue_frame, text="Queue cleared – jobs running…",
            font=("Segoe UI", 11), text_color="#64b5f6",
        ).pack(padx=8, pady=4)

    def run_queue():
        global _train_queue_running
        try:
            for qi, job in enumerate(jobs):
                output_queue.put(
                    f"\n{'─'*50}\n"
                    f"🏋  Queue job {qi + 1}/{len(jobs)}: {job['project_name']}\n"
                    f"{'─'*50}\n"
                )

                # Restore globals from job config
                global project_name, train_data_path, model_save_path, custom_model_path
                global input_size, epochs, batch_size, class_names, roboflow_yaml_path

                project_name      = job["project_name"]
                train_data_path   = job["train_data_path"]
                model_save_path   = job["model_save_path"]
                custom_model_path = job["custom_model_path"]
                input_size        = job["input_size"]
                epochs            = job["epochs"]
                batch_size        = job["batch_size"]
                class_names       = job["class_names"]
                roboflow_yaml_path = job["roboflow_yaml"]
                workers_int       = int(job.get("workers", 8))

                # Build YAML
                if job["roboflow_yaml"]:
                    yaml_path = job["roboflow_yaml"]
                else:
                    try:
                        yaml_path = create_yaml(
                            project_name, train_data_path, class_names, model_save_path
                        )
                    except Exception as exc:
                        output_queue.put(f"❌ Failed to create YAML: {exc}\n")
                        continue

                import json as _json
                _extra = job.get("extra_params", {})
                cmd = [
                    sys.executable, "src/train.py",
                    project_name,
                    train_data_path,
                    ",".join(class_names),
                    model_save_path,
                    job["selected_model"],
                    str(input_size),
                    str(epochs),
                    yaml_path,
                    str(batch_size),
                    custom_model_path,
                    _json.dumps(_extra),
                ]
                epoch_re = re.compile(r'^\s*(\d+)/(\d+)\s')

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                for raw_line in iter(proc.stdout.readline, ""):
                    line = _strip_ansi(raw_line)
                    if line.strip():
                        output_queue.put(line)
                    m = epoch_re.match(line)
                    if m:
                        ep_cur, ep_tot = int(m.group(1)), int(m.group(2))
                        root.after(0, lambda c=ep_cur, t=ep_tot: _update_queue_bar(c, t, qi, len(jobs)))
                proc.stdout.close()
                proc.wait()
                output_queue.put(f"\n✅  Job {qi + 1} complete: {project_name}\n")
        finally:
            _train_queue_running = False
            output_queue.put(f"\n🎉  All {len(jobs)} queued training job(s) finished.\n")
            root.after(0, lambda: progress_bar.set(1.0) if progress_bar else None)

    threading.Thread(target=run_queue, daemon=True).start()


def _update_queue_bar(ep_cur: int, ep_tot: int, job_idx: int, total_jobs: int) -> None:
    if progress_bar is None:
        return
    try:
        job_frac  = ep_cur / max(ep_tot, 1)
        total_frac = (job_idx + job_frac) / max(total_jobs, 1)
        progress_bar.set(total_frac)
        lbl = getattr(progress_bar, "_progress_label", None)
        if lbl:
            lbl.configure(
                text=(
                    f"Job {job_idx + 1}/{total_jobs} — "
                    f"Epoch {ep_cur}/{ep_tot}  ({total_frac * 100:.0f}%)"
                )
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Camera detection logic
# ─────────────────────────────────────────────────────────────────────────────
def start_camera_detection() -> None:
    global camera_detection, image_label, _camera_bar

    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt file) first.")
        return

    cam_text = camera_id_entry.get().strip() if camera_id_entry else ""
    if not cam_text:
        messagebox.showerror(
            "Error", "Please enter a Camera ID (e.g. 0 for the default webcam)."
        )
        return

    try:
        camera_id = int(cam_text)
    except ValueError:
        messagebox.showerror("Error", "Camera ID must be an integer (e.g. 0, 1, 2).")
        return

    if _camera_bar and hasattr(_camera_bar, "_start_btn"):
        _camera_bar._start_btn.configure(
            text="■  STOP",
            fg_color="#c62828",
            hover_color="#b71c1c",
            command=stop_camera_detection,
        )

    try:
        half = _camera_half_var.get() if _camera_half_var else False
        camera_detection = CameraDetection(detection_model_path, half=half, device=_get_device())
        camera_detection.start_camera(camera_id)
        if detection_save_dir:
            camera_detection.set_save_directory(detection_save_dir)
        camera_detection.show_camera_stream(image_label)
    except ValueError as exc:
        messagebox.showerror("Camera Error", f"Could not open camera {camera_id}:\n{exc}")
        _reset_camera_button()
    except Exception as exc:
        messagebox.showerror("Error", f"Unexpected error starting camera:\n{exc}")
        _reset_camera_button()


def stop_camera_detection() -> None:
    global camera_detection
    if camera_detection:
        camera_detection.stop()
        camera_detection = None
    _reset_camera_button()


def _reset_camera_button() -> None:
    if _camera_bar and hasattr(_camera_bar, "_start_btn"):
        _camera_bar._start_btn.configure(
            text="▶  START",
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=start_camera_detection,
        )


def save_callback() -> None:
    if camera_detection:
        if detection_save_dir:
            camera_detection.set_save_directory(detection_save_dir)
        camera_detection.capture_frame()


# ─────────────────────────────────────────────────────────────────────────────
#  GUI bootstrap
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

screen_w, screen_h = get_screen_size()
root = ctk.CTk()
root.title("YOLO Training & Detection Studio")

# Restore saved window size, falling back to full-screen dimensions
_app_cfg = _load_app_config()
_win_w = _app_cfg.get("window_width", screen_w)
_win_h = _app_cfg.get("window_height", screen_h)
root.geometry(f"{_win_w}x{_win_h}")


def _on_app_close() -> None:
    """Save window size to config, clean up temp files, then destroy the window."""
    try:
        cfg = _load_app_config()
        cfg["window_width"] = root.winfo_width()
        cfg["window_height"] = root.winfo_height()
        _save_app_config(cfg)
    except Exception:
        pass
    # Stop audio and remove any leftover temp audio file
    _cleanup_live_audio()
    # Remove the disk temp directory if it exists and is empty or was created by us
    try:
        import shutil as _shutil
        if _TEMP_DIR.exists():
            _shutil.rmtree(str(_TEMP_DIR), ignore_errors=True)
    except Exception:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", _on_app_close)

# ── Sidebar ────────────────────────────────────────────────────────────────────
SIDEBAR_W = 210
sidebar = ctk.CTkFrame(master=root, width=SIDEBAR_W, corner_radius=0, fg_color="#1e1e2e")
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

# Logo image – decoded from embedded base64 data
try:
    import base64 as _b64
    import io as _io
    _logo_data  = _b64.b64decode(_LOGO_B64)
    _logo_pil   = Image.open(_io.BytesIO(_logo_data)).resize((72, 72), Image.Resampling.LANCZOS)
    _logo_ctk   = ctk.CTkImage(light_image=_logo_pil, dark_image=_logo_pil, size=(72, 72))
    ctk.CTkLabel(sidebar, image=_logo_ctk, text="").pack(pady=(14, 4))
except Exception:
    pass  # logo is optional – skip silently if anything fails

ctk.CTkLabel(
    sidebar, text="YOLO Studio", font=("Segoe UI", 17, "bold"), text_color="#cdd6f4"
).pack(pady=(2, 2))
ctk.CTkLabel(
    sidebar, text="Train · Detect · Benchmark", font=("Segoe UI", 10), text_color="#6c7086"
).pack(pady=(0, 6))
ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=(0, 10))

_NAV = [
    ("🏋  Train",       "Train",      "#89b4fa"),
    ("🔍  Detect",      "Detect",     "#a6e3a1"),
    ("📷  Camera",      "Camera",     "#fab387"),
    ("🎬  Live Video",  "LiveVideo",  "#f2cdcd"),
    ("📊  Benchmark",   "Benchmark",  "#f9e2af"),
    ("⬇  Export",      "Export",     "#cba6f7"),
]
for _label, _key, _colour in _NAV:
    ctk.CTkButton(
        sidebar,
        text=_label,
        command=lambda k=_key: on_sidebar_select(k),
        fg_color=_colour,
        text_color="#1e1e2e",
        hover_color="#585b70",
        font=("Segoe UI", 14, "bold"),
        height=44,
        corner_radius=8,
    ).pack(fill="x", padx=10, pady=4)

ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=8)

# Appearance mode toggle
ctk.CTkLabel(
    sidebar, text="Appearance", font=("Segoe UI", 11), text_color="#a6adc8"
).pack(padx=10, anchor="w")
_appearance_var = ctk.StringVar(value="Dark")
ctk.CTkOptionMenu(
    sidebar,
    variable=_appearance_var,
    values=["Light", "Dark", "System"],
    command=ctk.set_appearance_mode,
    font=("Segoe UI", 11),
    height=30,
).pack(fill="x", padx=10, pady=(2, 4))

ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=(4, 4))

# ── Settings section ──────────────────────────────────────────────────────────
ctk.CTkLabel(
    sidebar, text="Settings", font=("Segoe UI", 11, "bold"), text_color="#a6adc8"
).pack(padx=10, pady=(2, 2), anchor="w")

_use_ram_temp_var = ctk.BooleanVar(value=True)
_ram_chk = ctk.CTkCheckBox(
    sidebar,
    text="Save temp files to RAM",
    variable=_use_ram_temp_var,
    font=("Segoe UI", 12),
)
_ram_chk.pack(fill="x", padx=10, pady=(2, 4))
Tooltip(
    _ram_chk,
    "When enabled, temporary files (such as extracted audio) are kept in memory\n"
    "rather than written to disk.  Faster and leaves no disk footprint.\n\n"
    "When disabled, all temporary files are stored in a 'temp/' folder next to\n"
    "the application and are cleaned up automatically.",
)

ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=(4, 4))

# ── CUDA / GPU section ────────────────────────────────────────────────────────
ctk.CTkLabel(
    sidebar, text="Hardware", font=("Segoe UI", 11, "bold"), text_color="#a6adc8"
).pack(padx=10, pady=(2, 2), anchor="w")

if _cuda_available:
    _cuda_label_text = f"CUDA ✅  Enabled"
    _cuda_label_color = "#4caf50"
    _cuda_detail = _cuda_device_name if _cuda_device_name else "GPU detected"
else:
    _cuda_label_text = "CUDA ❌  Not Detected"
    _cuda_label_color = "#ef5350"
    _cuda_detail = "CPU only"

ctk.CTkLabel(
    sidebar, text=_cuda_label_text,
    font=("Segoe UI", 12, "bold"), text_color=_cuda_label_color,
    anchor="center",
).pack(fill="x", padx=10)
ctk.CTkLabel(
    sidebar, text=_cuda_detail,
    font=("Segoe UI", 9), text_color="#6c7086",
    wraplength=180,
).pack(fill="x", padx=10, pady=(0, 4), anchor="w")

_gpu_device_var = ctk.BooleanVar(value=_cuda_available)  # default GPU when available

def _on_gpu_toggle():
    use_gpu = _gpu_device_var.get() and _cuda_available
    _gpu_toggle_lbl.configure(text="GPU" if use_gpu else "CPU")

_gpu_row = ctk.CTkFrame(sidebar, fg_color="transparent")
_gpu_row.pack(fill="x", padx=10, pady=(0, 4))
ctk.CTkLabel(_gpu_row, text="", width=1).pack(side="left", expand=True)  # left spacer
ctk.CTkLabel(_gpu_row, text="CPU", font=("Segoe UI", 12)).pack(side="left")
_gpu_switch = ctk.CTkSwitch(
    _gpu_row,
    text="",
    variable=_gpu_device_var,
    command=_on_gpu_toggle,
    state="normal" if _cuda_available else "disabled",
    width=44,
    height=22,
)
_gpu_switch.pack(side="left", padx=4)
_gpu_toggle_lbl = ctk.CTkLabel(
    _gpu_row,
    text="GPU" if _cuda_available else "CPU",
    font=("Segoe UI", 12),
    text_color="#4caf50" if _cuda_available else "#6c7086",
)
_gpu_toggle_lbl.pack(side="left")
ctk.CTkLabel(_gpu_row, text="", width=1).pack(side="left", expand=True)  # right spacer
Tooltip(
    _gpu_switch,
    "Switch between GPU (CUDA) and CPU inference.\n\n"
    "GPU is faster but requires a CUDA-capable NVIDIA graphics card.\n"
    "CPU mode always works but will be slower on large models.\n\n"
    f"{'CUDA is detected and ready.' if _cuda_available else 'CUDA is not available — CPU only.'}",
)

# Spacer + footer
ctk.CTkLabel(sidebar, text="").pack(fill="both", expand=True)
ctk.CTkLabel(
    sidebar, text="© 2026 PumpleX", font=("Segoe UI", 9), text_color="#585b70"
).pack(pady=6)

# ── Main frame ─────────────────────────────────────────────────────────────────
main_frame = ctk.CTkFrame(master=root, corner_radius=0)
main_frame.pack(fill="both", expand=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root.after(100, update_output_textbox)
    # Load the Train tab by default on startup
    root.after(200, lambda: on_sidebar_select("Train"))
    root.mainloop()
