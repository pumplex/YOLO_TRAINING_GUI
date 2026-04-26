# version 2.0.0 – benchmark tab, Roboflow ZIP import, segmentation models,
#                  custom model loader, TensorRT export, tooltips

import os
import sys
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
from src.train import create_yaml
from src.detect import detect_images, is_valid_image, get_model_info, get_media_files
from src.camera import CameraDetection

mimetypes.init()


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
_detect_progress_label    = None        # text progress label
_detect_nav_bar           = None        # bottom nav bar in detect view

# Live-video tab state
_live_video_path          = ""
_live_video_running       = False
_live_video_cancel_flag   = [False]
_live_video_label         = None
_live_video_status_label  = None
_live_video_start_btn     = None
_live_video_bar           = None

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
#  Output-queue consumer  (runs on main thread via root.after)
# ─────────────────────────────────────────────────────────────────────────────
def update_output_textbox() -> None:
    global output_textbox
    try:
        if output_textbox is not None and output_textbox.winfo_exists():
            line = output_queue.get_nowait()
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
    scale = min(max_w / img_w, max_h / img_h)
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
    global _detect_workers_var, _detect_progress_label, _detect_nav_bar
    global _live_video_label, _live_video_status_label, _live_video_start_btn, _live_video_bar
    global _train_queue_frame

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
    _detect_progress_label = None
    _detect_nav_bar = None
    _live_video_label = None
    _live_video_status_label = None
    _live_video_start_btn = None
    _live_video_bar = None
    _train_queue_frame = None

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
    options = DETECTION_MODELS if task_type_var.get() == "Detection" else SEGMENTATION_MODELS
    selected_model_var.set(options[0])
    model_menu_widget.configure(values=options)


def show_ai_train_window() -> None:
    global output_textbox, progress_bar, selected_model_var, task_type_var, model_menu_widget
    global train_data_label, model_save_label, custom_model_label

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

        _safe_label_configure(_rf_status_label, text="⏳ Extracting…", text_color="#64b5f6")
        config_panel.update_idletasks()

        try:
            from src.dataset import extract_roboflow_zip, count_dataset_images
            root_path, yaml, names = extract_roboflow_zip(zip_path, extract_dir)
        except Exception as exc:
            messagebox.showerror("Import Error", f"Failed to extract ZIP:\n{exc}")
            _safe_label_configure(_rf_status_label, text="Import failed.", text_color="#ef5350")
            return

        train_data_path    = root_path
        roboflow_yaml_path = yaml

        _safe_label_configure(train_data_label, text=Path(root_path).name, text_color="#64b5f6")

        # Auto-fill class names textbox
        class_names_text.delete("1.0", "end")
        class_names_text.insert("1.0", "\n".join(names))

        # Count images for a friendly status message
        try:
            from src.dataset import count_dataset_images
            counts = count_dataset_images(root_path)
            parts  = [f"{v} {k}" for k, v in counts.items()]
            count_str = "  |  ".join(parts) if parts else "unknown"
        except Exception:
            count_str = "unknown"

        status = f"✅  {Path(zip_path).stem}  •  {len(names)} classes  •  {count_str} images"
        _safe_label_configure(_rf_status_label, text=status, text_color="#4caf50")

        preview = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
        messagebox.showinfo(
            "Dataset Imported Successfully",
            f"Roboflow dataset extracted to:\n{root_path}\n\n"
            f"Classes ({len(names)}): {preview}\n\n"
            f"Images: {count_str}\n\n"
            "Class names have been filled in automatically.\n"
            "Select a model, set epochs/batch, and click Start Training.",
        )

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
    Tooltip(
        train_data_btn,
        "Select a folder containing paired image + YOLO annotation (.txt) files.\n\n"
        "Expected layout:\n"
        "  folder/\n"
        "    photo1.jpg   photo1.txt\n"
        "    photo2.png   photo2.txt  …\n\n"
        "The app will automatically split 80 % → train, 20 % → val.\n\n"
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
    task_frame = ctk.CTkFrame(config_panel, fg_color="transparent")
    task_frame.pack(fill="x", **PAD)
    ctk.CTkRadioButton(
        task_frame, text="Detection",
        variable=task_type_var, value="Detection",
        command=_on_task_type_change, font=FLAB,
    ).pack(side="left", padx=(0, 24))
    ctk.CTkRadioButton(
        task_frame, text="Segmentation",
        variable=task_type_var, value="Segmentation",
        command=_on_task_type_change, font=FLAB,
    ).pack(side="left")
    Tooltip(
        task_frame,
        "Detection     – predicts bounding boxes around objects.\n"
        "Segmentation  – predicts pixel-level instance masks.\n\n"
        "Segmentation requires polygon annotations in your dataset.",
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
    ctk.CTkButton(
        config_panel, text="Browse .pt…", font=FBTN, height=36, command=select_custom_model
    ).pack(fill="x", **PAD)
    Tooltip(
        config_panel.winfo_children()[-1],
        "Load your own .pt file as the training starting point.\n"
        "When set, this overrides the YOLO Model dropdown above.\n\n"
        "Useful for fine-tuning an already-trained custom model.",
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
    input_size_entry.pack(fill="x", **PAD)
    Tooltip(
        input_size_entry,
        "Square resolution fed into the network (pixels).\n"
        "640 is standard.  Use 416 for faster training or\n"
        "1280 for higher precision on large images.",
    )

    _lbl("Epochs  (e.g. 100)")
    epochs_entry = ctk.CTkEntry(
        config_panel, placeholder_text="100", font=FENT, height=36
    )
    epochs_entry.pack(fill="x", **PAD)
    Tooltip(
        epochs_entry,
        "Number of full passes over the training dataset.\n"
        "More epochs → longer training, potentially better accuracy.\n"
        "Start with 50–100; increase if validation loss is still improving.",
    )

    _lbl("Batch Size  (e.g. 16)")
    batch_size_entry = ctk.CTkEntry(
        config_panel, placeholder_text="16", font=FENT, height=36
    )
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
    workers_entry.pack(fill="x", **PAD)
    Tooltip(
        workers_entry,
        "Number of CPU worker threads used to load data during training.\n"
        "Higher values can speed up data loading on multi-core machines.\n"
        "Reduce to 0 on Windows if you see multiprocessing errors.",
    )
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
        "If you imported a Roboflow ZIP, these are filled automatically.",
    )
    _sep()

    # ── Training queue ─────────────────────────────────────────────────────
    _lbl("Training Queue")

    def _get_current_job_config():
        pname   = project_name_entry.get().strip()
        isize   = input_size_entry.get().strip() or "640"
        ep      = epochs_entry.get().strip() or "100"
        bs      = batch_size_entry.get().strip() or "16"
        wk      = workers_entry.get().strip() or "8"
        raw_cls = class_names_text.get("1.0", "end-1c")
        cls     = [n.strip() for n in raw_cls.splitlines() if n.strip()]
        sel_disp  = selected_model_var.get() if selected_model_var else ""
        sel_model = MODEL_MAP.get(sel_disp, "")
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

    # ── Start / Queue-Run buttons ──────────────────────────────────────────
    btn_row = ctk.CTkFrame(config_panel, fg_color="transparent")
    btn_row.pack(fill="x", padx=14, pady=(4, 12))

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

    ctk.CTkButton(
        btn_row,
        text="▶▶  Run Queue",
        command=_run_training_queue,
        fg_color="#4a148c",
        hover_color="#311b92",
        font=("Segoe UI", 14, "bold"),
        height=46,
        text_color="white",
        corner_radius=8,
    ).pack(side="left", fill="x", expand=True, padx=(4, 0))
    Tooltip(
        btn_row.winfo_children()[1],
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

    train_progress_label = ctk.CTkLabel(
        log_panel, text="", font=("Segoe UI", 11), text_color="#a6adc8", anchor="w",
    )
    train_progress_label.pack(anchor="w", padx=12)

    progress_bar = ctk.CTkProgressBar(
        log_panel, progress_color="#43a047", mode="determinate",
    )
    progress_bar.set(0)
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
    global _detect_conf_var, _detect_half_var, _detect_workers_var
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
        cfg, text="No folder selected", font=("Segoe UI", 10),
        text_color="gray", anchor="w", wraplength=220,
    )
    detect_folder_label.pack(fill="x", padx=12)

    _detect_file_count_label = ctk.CTkLabel(
        cfg, text="", font=("Segoe UI", 10), text_color="#6c7086", anchor="w",
    )
    _detect_file_count_label.pack(fill="x", padx=12)
    _csep()

    # Model section
    _clbl("🤖  YOLO Model (.pt)")
    sel_model_btn = ctk.CTkButton(
        cfg, text="Browse Model…", font=FBTN, height=34,
        command=select_detection_model,
    )
    sel_model_btn.pack(fill="x", padx=12, pady=(4, 2))
    Tooltip(sel_model_btn, "Choose a trained YOLO .pt weights file for inference.")

    detect_model_label = ctk.CTkLabel(
        cfg, text="No model selected", font=("Segoe UI", 10),
        text_color="gray", anchor="w", wraplength=220,
    )
    detect_model_label.pack(fill="x", padx=12)

    _detect_model_info_label = ctk.CTkLabel(
        cfg, text="", font=("Segoe UI", 10), text_color="#6c7086",
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
                        workers_entry_d]

    # ── Right: image viewer ────────────────────────────────────────────────
    viewer = ctk.CTkFrame(main_frame, corner_radius=8, fg_color="#0d1117")
    viewer.place(relx=0.29, rely=0, relwidth=0.71, relheight=1.0)

    image_label = tk.Label(viewer, bg="#0d1117")
    image_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.88)

    # Bottom navigation bar
    _detect_nav_bar = ctk.CTkFrame(viewer, corner_radius=0, fg_color="#1e1e2e", height=50)
    _detect_nav_bar.place(relx=0, rely=0.88, relwidth=1.0, relheight=0.12)

    ctk.CTkButton(
        _detect_nav_bar, text="◀", command=show_prev_image,
        fg_color="#1976d2", font=("Segoe UI", 18, "bold"), height=32, width=40,
    ).place(relx=0.01, rely=0.1, relwidth=0.08, relheight=0.8)

    image_index_label = ctk.CTkLabel(_detect_nav_bar, text="No results", font=("Segoe UI", 13))
    image_index_label.place(relx=0.10, rely=0.1, relwidth=0.20, relheight=0.8)

    ctk.CTkButton(
        _detect_nav_bar, text="▶", command=show_next_image,
        fg_color="#1976d2", font=("Segoe UI", 18, "bold"), height=32, width=40,
    ).place(relx=0.31, rely=0.1, relwidth=0.08, relheight=0.8)

    ctk.CTkButton(
        _detect_nav_bar, text="⛶  Full Screen", command=_open_fullscreen_image,
        fg_color="#37474f", hover_color="#263238",
        font=("Segoe UI", 12), height=32,
    ).place(relx=0.42, rely=0.1, relwidth=0.18, relheight=0.8)
    Tooltip(_detect_nav_bar.winfo_children()[-1], "Open the current image in a full-screen viewer.")

    _detect_progress_label = ctk.CTkLabel(
        _detect_nav_bar, text="", font=("Segoe UI", 11), text_color="#a6adc8", anchor="w",
    )
    _detect_progress_label.place(relx=0.62, rely=0.05, relwidth=0.37, relheight=0.5)

    detection_progress_bar = ctk.CTkProgressBar(
        _detect_nav_bar, progress_color="#43a047", mode="determinate",
    )
    detection_progress_bar.set(0)
    detection_progress_bar.place(relx=0.62, rely=0.55, relwidth=0.37, relheight=0.35)


# ─────────────────────────────────────────────────────────────────────────────
#  Camera Detection window
# ─────────────────────────────────────────────────────────────────────────────
def show_camera_detection_window() -> None:
    global camera_detection, camera_id_entry, image_label, _camera_bar

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
    sel_model_btn.place(relx=0.01, rely=0.1, relwidth=0.14, relheight=0.8)
    Tooltip(sel_model_btn, "Choose the YOLO .pt model file used for live camera inference.")

    sel_save_btn = ctk.CTkButton(
        bar, text="Save Folder",
        command=select_camera_save_folder, font=FONT, height=34,
    )
    sel_save_btn.place(relx=0.17, rely=0.1, relwidth=0.11, relheight=0.8)
    Tooltip(sel_save_btn, "Folder where captured frames are saved when you press Enter.")

    camera_id_entry = ctk.CTkEntry(
        bar, placeholder_text="Camera ID  (e.g. 0)", font=FONT, height=34
    )
    camera_id_entry.place(relx=0.30, rely=0.1, relwidth=0.15, relheight=0.8)
    Tooltip(
        camera_id_entry,
        "Index of the camera to open.\n"
        "0 = default webcam, 1 = second camera, etc.",
    )

    ctk.CTkLabel(
        bar,
        text="Press  Enter  to capture & save a frame",
        font=("Segoe UI", 11),
        text_color="gray",
    ).place(relx=0.48, rely=0.1, relwidth=0.28, relheight=0.8)

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

    _live_video_path = ""

    # Display area
    _live_video_label = tk.Label(main_frame, bg="#0d0d0d")
    _live_video_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.91)

    bar = ctk.CTkFrame(main_frame, corner_radius=0, height=56)
    bar.place(relx=0, rely=0.91, relwidth=1.0, relheight=0.09)
    _live_video_bar = bar

    FONT = ("Segoe UI", 12)
    _video_path_lbl = ctk.CTkLabel(bar, text="No video selected", font=("Segoe UI", 11),
                                   text_color="gray", anchor="w")
    _video_path_lbl.place(relx=0.01, rely=0.05, relwidth=0.52, relheight=0.45)

    _live_video_status_label = ctk.CTkLabel(bar, text="", font=("Segoe UI", 11),
                                            text_color="#a6adc8", anchor="w")
    _live_video_status_label.place(relx=0.01, rely=0.55, relwidth=0.52, relheight=0.42)

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
            _video_path_lbl.configure(text=Path(p).name, text_color="#4caf50")

    vid_btn = ctk.CTkButton(bar, text="📂  Select Video", command=_pick_video,
                             font=FONT, height=34)
    vid_btn.place(relx=0.55, rely=0.1, relwidth=0.14, relheight=0.8)
    Tooltip(vid_btn, "Choose a local video file to play back with live YOLO detection overlay.")

    model_btn = ctk.CTkButton(bar, text="🤖  Select Model",
                               command=select_detection_model, font=FONT, height=34)
    model_btn.place(relx=0.71, rely=0.1, relwidth=0.13, relheight=0.8)
    Tooltip(model_btn, "Choose the YOLO .pt model used for detection on the video frames.")

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
        font=("Segoe UI", 14, "bold"), height=34, text_color="white",
    )
    _live_video_start_btn.place(relx=0.86, rely=0.1, relwidth=0.13, relheight=0.8)
    bar._start_btn = _live_video_start_btn


def _start_live_video() -> None:
    global _live_video_running, _live_video_cancel_flag

    if not _live_video_path:
        messagebox.showerror("Error", "Please select a video file first.")
        return
    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt) first.")
        return

    _live_video_running = True
    _live_video_cancel_flag[0] = False

    if _live_video_start_btn:
        _live_video_start_btn.configure(
            text="■  STOP", fg_color="#c62828", hover_color="#b71c1c",
        )

    threading.Thread(target=_live_video_thread, daemon=True).start()


def _stop_live_video() -> None:
    global _live_video_running
    _live_video_running = False
    _live_video_cancel_flag[0] = True
    if _live_video_start_btn:
        _live_video_start_btn.configure(
            text="▶  PLAY", fg_color="#2e7d32", hover_color="#1b5e20",
        )


def _live_video_thread() -> None:
    """Background thread: open video, run YOLO frame-by-frame, display in label."""
    import re
    try:
        from ultralytics import YOLO as _YOLO
        model = _YOLO(detection_model_path)

        cap = cv2.VideoCapture(_live_video_path)
        if not cap.isOpened():
            root.after(0, lambda: messagebox.showerror("Error", "Could not open video file."))
            _stop_live_video()
            return

        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = max(1, int(1000 / fps))
        frame_idx = 0

        while not _live_video_cancel_flag[0]:
            ret, frame = cap.read()
            if not ret:
                break  # video ended – stop

            results = model.predict(frame, save=False, conf=0.5, verbose=False)
            annotated = results[0].plot()

            img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Resize to fit label
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
            pct = frame_idx / total_frames * 100
            status = f"Frame {frame_idx}/{total_frames}  ({pct:.0f}%)  |  {det_count} detection(s)"

            def _update(ph=photo, st=status):
                if _live_video_label:
                    try:
                        _live_video_label.config(image=ph)
                        _live_video_label.image = ph
                    except Exception:
                        pass
                _safe_label_configure(_live_video_status_label, text=st)

            root.after(0, _update)
            frame_idx += 1

            import time
            time.sleep(max(0.001, delay / 1000 - 0.01))

        cap.release()
    except Exception as exc:
        root.after(0, lambda: messagebox.showerror("Live Video Error", str(exc)))
    finally:
        global _live_video_running
        _live_video_running = False
        root.after(0, lambda: _stop_live_video() if _live_video_start_btn else None)


# ─────────────────────────────────────────────────────────────────────────────
#  Export window
# ─────────────────────────────────────────────────────────────────────────────
def show_export_window() -> None:
    global export_model_label, export_status_label, export_model_path
    export_model_path = ""

    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)

    ctk.CTkLabel(
        main_frame, text="Export Trained Model",
        font=("Segoe UI", 20, "bold"),
    ).place(relx=0.5, rely=0.06, anchor="center")

    ctk.CTkLabel(main_frame, text="Trained model (.pt)", font=FLAB).place(
        relx=0.25, rely=0.14, anchor="center"
    )
    sel_btn = ctk.CTkButton(
        main_frame, text="Browse .pt…", font=FBTN, height=38,
        command=select_export_model,
    )
    sel_btn.place(relx=0.25, rely=0.21, anchor="center", relwidth=0.30)
    Tooltip(sel_btn, "Select the trained YOLO .pt model you want to export.")

    export_model_label = ctk.CTkLabel(
        main_frame, text="No model selected", font=("Segoe UI", 11), text_color="gray",
    )
    export_model_label.place(relx=0.25, rely=0.28, anchor="center", relwidth=0.42)

    ctk.CTkLabel(main_frame, text="Export Format", font=FLAB).place(
        relx=0.25, rely=0.36, anchor="center"
    )
    export_fmt_var = ctk.StringVar(value=EXPORT_FORMATS[0])
    fmt_menu = ctk.CTkOptionMenu(
        main_frame, variable=export_fmt_var, values=EXPORT_FORMATS,
        font=FBTN, height=38,
    )
    fmt_menu.place(relx=0.25, rely=0.43, anchor="center", relwidth=0.30)
    Tooltip(
        fmt_menu,
        "ONNX            – universal format; runs on CPU, GPU, or accelerators.\n"
        "TensorRT Engine – maximum throughput on NVIDIA GPUs; device-specific.\n"
        "CoreML          – Apple devices (macOS / iOS).\n"
        "TF SavedModel   – TensorFlow ecosystem.\n"
        "TFLite          – mobile / embedded TensorFlow.",
    )

    _trt_note = (
        "ℹ️  TensorRT notes\n\n"
        "Exporting to a TensorRT .engine file compiles the model into GPU-specific\n"
        "machine code for maximum inference speed on NVIDIA hardware.\n\n"
        "Requirements:\n"
        "  • NVIDIA GPU with CUDA ≥ 11\n"
        "  • TensorRT ≥ 8  (pip install tensorrt)\n"
        "  • The .engine file is bound to the GPU it was built on.\n\n"
        "This is an inference-optimisation step, NOT a training format."
    )
    note_box = ctk.CTkTextbox(
        main_frame, font=("Segoe UI", 11), height=145,
        fg_color="#2d2d1e", text_color="#e8e8b0", corner_radius=8,
    )
    note_box.place(relx=0.68, rely=0.38, anchor="center", relwidth=0.54)
    note_box.insert("1.0", _trt_note)
    note_box.configure(state="disabled")

    ctk.CTkButton(
        main_frame,
        text="⬇  Export Model",
        command=lambda: export_model(export_fmt_var.get()),
        fg_color="#6a1b9a", hover_color="#4a148c",
        font=("Segoe UI", 15, "bold"), height=50,
        text_color="white", corner_radius=8,
    ).place(relx=0.25, rely=0.58, anchor="center", relwidth=0.32)

    export_status_label = ctk.CTkLabel(
        main_frame, text="", font=("Segoe UI", 12), wraplength=600
    )
    export_status_label.place(relx=0.5, rely=0.72, anchor="center", relwidth=0.85)


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
            title="Select YOLO model(s)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
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
        "Add one or more trained .pt model files to compare.\n"
        "You can benchmark as many models as you like side-by-side.",
    )
    _sep()

    # ── Dataset YAML ──────────────────────────────────────────────────────
    _lbl("Dataset YAML")
    _yaml_ref = [""]  # mutable container captured by closure

    yaml_lbl = ctk.CTkLabel(
        setup, text="No YAML selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )

    def _select_yaml():
        p = normalize_path(
            filedialog.askopenfilename(
                title="Select dataset YAML",
                filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            )
        )
        if p:
            _yaml_ref[0] = p
            yaml_lbl.configure(text=Path(p).name, text_color="#4caf50")

    yaml_btn = ctk.CTkButton(setup, text="Browse YAML…", font=FBTN, height=36, command=_select_yaml)
    yaml_btn.pack(fill="x", **PAD)
    Tooltip(
        yaml_btn,
        "Select the data.yaml for your dataset.\n\n"
        "This YAML tells YOLO where your validation / test images and labels are.\n"
        "Roboflow-exported datasets include data.yaml in the ZIP root.\n\n"
        "The same YAML can be used to compare multiple models.",
    )
    yaml_lbl.pack(fill="x", padx=14)
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
    split_var = ctk.StringVar(value="val")
    split_frame = ctk.CTkFrame(setup, fg_color="transparent")
    split_frame.pack(fill="x", **PAD)
    for _txt, _val in [("Validation set", "val"), ("Test set", "test")]:
        ctk.CTkRadioButton(
            split_frame, text=_txt, variable=split_var, value=_val, font=FLAB,
        ).pack(side="left", padx=(0, 20))
    Tooltip(
        split_frame,
        "Validation set  – the val split used during training (always available).\n"
        "Test set        – a held-out set never seen during training.\n"
        "                  Only available if your dataset has a 'test' split.",
    )
    _sep()

    # ── Run button ────────────────────────────────────────────────────────
    _benchmark_run_btn = ctk.CTkButton(
        setup,
        text="▶  Run Benchmark",
        fg_color="#1565c0", hover_color="#0d47a1",
        font=("Segoe UI", 15, "bold"), height=50,
        text_color="white", corner_radius=8,
        command=lambda: _start_benchmark(img_size_entry, split_var, _yaml_ref),
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
            "Add your trained models and a dataset YAML on the left,\n"
            "then click  ▶ Run Benchmark  to compare them side-by-side.\n\n"
            "Results will show accuracy, speed, and model size —\n"
            "top performers highlighted in green / blue."
        ),
        font=("Segoe UI", 13),
        text_color="gray",
        justify="center",
    ).pack(pady=4)


def _start_benchmark(img_size_entry, split_var, yaml_ref) -> None:
    global _benchmark_results_frame, _benchmark_run_btn

    yaml_path = yaml_ref[0]
    img_size_str = img_size_entry.get().strip() or "640"

    errors = []
    if not _benchmark_models:
        errors.append("• Please add at least one model.")
    if not yaml_path:
        errors.append("• Please select a dataset YAML file.")
    if not img_size_str.isdigit() or int(img_size_str) < 1:
        errors.append("• Image Size must be a positive integer (e.g. 640).")
    if errors:
        messagebox.showerror("Missing input", "\n".join(errors))
        return

    img_size   = int(img_size_str)
    split      = split_var.get()
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
        from ultralytics import YOLO
        all_metrics = []
        for i, mp in enumerate(models_run):
            _log(f"\n[{i + 1}/{total_m}]  Evaluating:  {Path(mp).name}\n")
            _set_progress(i)
            try:
                model  = YOLO(mp)

                # Pre-flight: check class count compatibility
                yaml_nc = _get_yaml_nc(yaml_path)
                model_nc = len(model.names) if hasattr(model, "names") else None
                if yaml_nc is not None and model_nc is not None and model_nc != yaml_nc:
                    raise ValueError(
                        f"Class count mismatch: model has {model_nc} classes but "
                        f"dataset YAML declares {yaml_nc} classes.  "
                        "Benchmark skipped for this model."
                    )

                result = model.val(data=yaml_path, imgsz=img_size, split=split, verbose=False)
                m      = _extract_bench_metrics(mp, result)
                all_metrics.append(m)
                _log(
                    f"  mAP50={m['map50']:.3f}  mAP50-95={m['map']:.3f}  "
                    f"Speed={m['speed_ms']:.1f} ms/img\n"
                )
            except Exception as exc:
                friendly = _classify_error(exc)
                _log(f"  ❌  Error: {friendly}\n")
                all_metrics.append({
                    "name": Path(mp).name, "path": mp,
                    "map50": None, "map": None,
                    "precision": None, "recall": None,
                    "speed_ms": None,
                    "size_mb": Path(mp).stat().st_size / 1_048_576,
                    "error": str(exc),
                })
        _set_progress(total_m)
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
    ).pack(anchor="w", padx=16, pady=(2, 8))


# ─────────────────────────────────────────────────────────────────────────────
#  File / folder selection dialogs
# ─────────────────────────────────────────────────────────────────────────────
def select_train_data() -> None:
    global train_data_path, train_data_label
    path = normalize_path(filedialog.askdirectory(title="Select Training Data Folder"))
    if path:
        train_data_path = path
        short = Path(path).name or path
        _safe_label_configure(train_data_label, text=short, text_color="#4caf50")


def select_model_save_folder() -> None:
    global model_save_path, model_save_label
    path = normalize_path(filedialog.askdirectory(title="Select Model Save Folder"))
    if path:
        model_save_path = path
        short = Path(path).name or path
        _safe_label_configure(model_save_label, text=short, text_color="#4caf50")


def select_custom_model() -> None:
    global custom_model_path, custom_model_label
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select Custom Base Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
    )
    if path:
        custom_model_path = path
        _safe_label_configure(
            custom_model_label,
            text=f"Custom: {Path(path).name}",
            text_color="#64b5f6",
        )


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

    # Count files in background (can be slow for large folders)
    def _count():
        try:
            imgs, vids = get_media_files(path)
            txt = f"{len(imgs)} image(s),  {len(vids)} video(s)"
        except Exception:
            txt = "Could not count files"
        root.after(0, lambda: _safe_label_configure(_detect_file_count_label, text=txt))

    threading.Thread(target=_count, daemon=True).start()
    _safe_label_configure(_detect_file_count_label, text="Counting files…")


def select_detection_model() -> None:
    global detection_model_path, detect_model_label, image_paths, current_image_index
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
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
        except Exception as exc:
            txt = f"Could not load info: {exc}"
        root.after(
            0,
            lambda: _safe_label_configure(_detect_model_info_label, text=txt, text_color="#a6adc8"),
        )

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
        errors.append("• Epochs must be a positive integer (e.g. 100).")
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
    workers_int = int(workers_val) if workers_val.isdigit() else 8

    # Use the Roboflow YAML directly if one was imported, otherwise build one
    if roboflow_yaml_path:
        yaml_path = roboflow_yaml_path
    else:
        yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)

    _run_training_subprocess(yaml_path, selected_model_size, workers_int)


def _run_training_subprocess(
    yaml_path: str, selected_model_size: str, workers_int: int = 8
) -> None:
    global progress_bar, output_textbox
    import re as _re

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
    ]

    total_epochs = int(epochs) if str(epochs).isdigit() else 1
    # Pattern: lines like "      1/100  " at the start
    epoch_re = _re.compile(r'^\s*(\d+)/(\d+)\s')

    def _update_train_progress(current_ep: int, total_ep: int) -> None:
        if progress_bar is None:
            return
        try:
            if not progress_bar.winfo_exists():
                return
            frac = current_ep / max(total_ep, 1)
            progress_bar.set(frac)
            lbl = getattr(progress_bar, "_progress_label", None)
            if lbl:
                try:
                    lbl.configure(text=f"Epoch {current_ep} / {total_ep}  ({frac * 100:.0f}%)")
                except Exception:
                    pass
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

        def _reader():
            for line in iter(proc.stdout.readline, ""):
                output_queue.put(line)
                m = epoch_re.match(line)
                if m:
                    ep_cur = int(m.group(1))
                    ep_tot = int(m.group(2))
                    root.after(0, lambda c=ep_cur, t=ep_tot: _update_train_progress(c, t))
            proc.stdout.close()

        threading.Thread(target=_reader, daemon=True).start()
        proc.wait()
        root.after(0, _training_finished)

    if progress_bar:
        try:
            progress_bar.set(0)
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def _training_finished() -> None:
    global progress_bar, output_textbox
    if progress_bar:
        try:
            progress_bar.set(1.0)
            lbl = getattr(progress_bar, "_progress_label", None)
            if lbl:
                lbl.configure(text="Training complete ✅")
        except Exception:
            pass
    try:
        if output_textbox and output_textbox.winfo_exists():
            output_textbox.insert("end", "\n✅ Training process finished.\n")
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

    def _progress_cb(current: int, total: int, msg: str) -> None:
        frac = current / max(total, 1)
        root.after(0, lambda: _update_detect_progress(frac, current, total, msg))

    threading.Thread(
        target=detect_images,
        kwargs=dict(
            images_folder=detection_images_folder_path,
            model_path=detection_model_path,
            callback=_on_detection_complete,
            progress_callback=_progress_cb,
            conf_threshold=conf,
            half=half,
            workers=workers_val,
            cancel_flag=lambda: _detection_cancel_flag[0],
        ),
        daemon=True,
    ).start()


def _cancel_image_detection() -> None:
    global _detection_cancel_flag
    _detection_cancel_flag[0] = True
    _safe_label_configure(_detect_progress_label, text="Cancelling…")


def _update_detect_progress(frac: float, current: int, total: int, msg: str) -> None:
    if detection_progress_bar:
        try:
            detection_progress_bar.set(frac)
        except Exception:
            pass
    pct_txt = f"{frac * 100:.0f}%  ({current}/{total})  {msg}"
    _safe_label_configure(_detect_progress_label, text=pct_txt)
    _safe_label_configure(image_index_label, text=f"{current}/{total}")


def _on_detection_complete(results_dir: str) -> None:
    global image_paths, current_image_index
    image_paths = sorted(
        str(p) for p in Path(results_dir).iterdir()
        if p.is_file() and is_valid_image(str(p))
    )
    current_image_index = 0
    root.after(0, _show_detection_results)


def _show_detection_results() -> None:
    global _detection_running
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

    if detection_progress_bar:
        try:
            detection_progress_bar.set(1.0 if image_paths else 0.0)
        except Exception:
            pass

    if image_paths:
        n = len(image_paths)
        _safe_label_configure(_detect_progress_label, text=f"Done – {n} result image(s)")
        _safe_label_configure(image_index_label, text=f"1/{n}")
        update_image()
    else:
        _safe_label_configure(_detect_progress_label, text="No result images found")
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
                ]
                import re as _re
                epoch_re = _re.compile(r'^\s*(\d+)/(\d+)\s')
                total_ep = int(epochs) if str(epochs).isdigit() else 1

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                for line in iter(proc.stdout.readline, ""):
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
        camera_detection = CameraDetection(detection_model_path)
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
#  Export logic
# ─────────────────────────────────────────────────────────────────────────────
def export_model(format_display: str) -> None:
    global export_status_label, export_model_path

    if not export_model_path:
        messagebox.showerror("Error", "Please select a trained .pt model to export.")
        return

    fmt = EXPORT_FORMAT_MAP.get(format_display, "onnx")
    _safe_label_configure(
        export_status_label,
        text=f"Exporting to {format_display}…  please wait.",
        text_color="#64b5f6",
    )
    root.update()

    def do_export() -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(export_model_path)
            out = model.export(format=fmt)
            msg = f"✅ Export successful →  {out}"
            root.after(
                0,
                lambda: _safe_label_configure(export_status_label, text=msg, text_color="#4caf50"),
            )
        except Exception as exc:
            err = f"❌ Export failed: {exc}"
            root.after(
                0,
                lambda: _safe_label_configure(export_status_label, text=err, text_color="#ef5350"),
            )

    threading.Thread(target=do_export, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  GUI bootstrap
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

screen_w, screen_h = get_screen_size()
root = ctk.CTk()
root.title("YOLO Training & Detection Studio")
root.geometry(f"{screen_w}x{screen_h}")

# ── Sidebar ────────────────────────────────────────────────────────────────────
SIDEBAR_W = 210
sidebar = ctk.CTkFrame(master=root, width=SIDEBAR_W, corner_radius=0, fg_color="#1e1e2e")
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

ctk.CTkLabel(
    sidebar, text="YOLO Studio", font=("Segoe UI", 17, "bold"), text_color="#cdd6f4"
).pack(pady=(18, 2))
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
).pack(fill="x", padx=10, pady=(2, 8))

# Spacer + footer
ctk.CTkLabel(sidebar, text="").pack(fill="both", expand=True)
ctk.CTkLabel(
    sidebar, text="© 2024 SpreadKnowledge", font=("Segoe UI", 9), text_color="#585b70"
).pack(pady=6)

# ── Main frame ─────────────────────────────────────────────────────────────────
main_frame = ctk.CTkFrame(master=root, corner_radius=0)
main_frame.pack(fill="both", expand=True)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root.after(100, update_output_textbox)
    root.mainloop()
