import os
import sys
import torch
import shutil
import glob
import random
import mimetypes
from pathlib import Path
from ultralytics import YOLO

# ── Centralised models cache ──────────────────────────────────────────────────
# All pre-trained weights are stored in a "models" folder next to this script's
# parent directory so they are only downloaded once.
_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = _SCRIPT_DIR.parent
_MODELS_DIR = _APP_DIR / "models"
_MODELS_DIR.mkdir(exist_ok=True)

try:
    from ultralytics import settings as _ult_settings
    _ult_settings.update({"weights_dir": str(_MODELS_DIR)})
except Exception:
    pass

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

def is_valid_image(file_path):
    """Check if a file is a valid image by examining extension and mime type"""
    try:
        file_path = Path(file_path)
        
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_IMAGE_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type is not None and mime_type.startswith('image/')
    except Exception:
        return False

def normalize_path(path):
    if not path:
        return path
    return str(Path(path).resolve())

def _find_paired_files_in_dir(images_dir: Path, labels_dir: Path) -> list:
    """Return a list of (img_name, lbl_name) pairs from separate images/labels directories."""
    paired = []
    for img_path in images_dir.iterdir():
        if img_path.is_file() and is_valid_image(str(img_path)):
            lbl_path = labels_dir / img_path.with_suffix('.txt').name
            if lbl_path.exists():
                paired.append((img_path.name, lbl_path.name))
    return paired


def prepare_data(train_data_path):
    train_data_path = normalize_path(train_data_path)
    train_path = Path(train_data_path)

    # ── Case 1: Already has lowercase train/val split ─────────────────────────
    train_dir_exists = (
        (train_path / 'train/images').exists()
        and (train_path / 'train/labels').exists()
    )
    val_dir_exists = (
        (train_path / 'val/images').exists()
        and (train_path / 'val/labels').exists()
    )
    if train_dir_exists and val_dir_exists:
        print("Train and validation directories already exist. Skipping file preparation.")
        return

    # ── Case 2: Sub-folder split (e.g. Train/images, Valid/images) ────────────
    # Accept any capitalisation of: train, valid, val, test
    _SPLIT_ALIASES = {
        'train': ['train', 'Train', 'TRAIN'],
        'val':   ['val', 'Val', 'VAL', 'valid', 'Valid', 'VALID', 'validation', 'Validation'],
        'test':  ['test', 'Test', 'TEST'],
    }
    found_splits = {}  # canonical_name → (images_dir, labels_dir)
    for canonical, aliases in _SPLIT_ALIASES.items():
        for alias in aliases:
            img_d = train_path / alias / 'images'
            lbl_d = train_path / alias / 'labels'
            if img_d.exists() and lbl_d.exists():
                found_splits[canonical] = (img_d, lbl_d)
                break

    if 'train' in found_splits:
        # Normalise by moving files into the expected lowercase train/val layout
        for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
            (train_path / path).mkdir(parents=True, exist_ok=True)

        for canonical, (img_d, lbl_d) in found_splits.items():
            dest_key = canonical if canonical in ('train', 'val') else 'train'
            dst_img = train_path / dest_key / 'images'
            dst_lbl = train_path / dest_key / 'labels'
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for f in img_d.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(dst_img / f.name))
            for f in lbl_d.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(dst_lbl / f.name))

        # If no val split was found, fall back to splitting the train set
        if 'val' not in found_splits:
            _split_images_labels(
                train_path / 'train/images',
                train_path / 'train/labels',
                train_path,
            )
        print("Normalised sub-folder split layout. Preparation complete.")
        return

    # ── Case 3: Flat images/ + labels/ sub-directories (no split) ─────────────
    flat_imgs = train_path / 'images'
    flat_lbls = train_path / 'labels'
    if flat_imgs.exists() and flat_lbls.exists():
        for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
            (train_path / path).mkdir(parents=True, exist_ok=True)

        paired_files = _find_paired_files_in_dir(flat_imgs, flat_lbls)
        random.seed(0)
        random.shuffle(paired_files)
        split_idx = int(len(paired_files) * 0.8)

        for img_name, lbl_name in paired_files[:split_idx]:
            shutil.move(str(flat_imgs / img_name), str(train_path / 'train/images' / img_name))
            shutil.move(str(flat_lbls / lbl_name), str(train_path / 'train/labels' / lbl_name))
        for img_name, lbl_name in paired_files[split_idx:]:
            shutil.move(str(flat_imgs / img_name), str(train_path / 'val/images' / img_name))
            shutil.move(str(flat_lbls / lbl_name), str(train_path / 'val/labels' / lbl_name))
        print("Prepared data from flat images/labels directories. Preparation complete.")
        return

    # ── Case 4: Flat files in root (image + .txt pairs side by side) ──────────
    for path in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        (train_path / path).mkdir(parents=True, exist_ok=True)

    paired_files = []
    for file_path in Path(train_data_path).iterdir():
        if file_path.is_file() and is_valid_image(str(file_path)):
            txt_file = file_path.with_suffix('.txt')
            if txt_file.exists():
                paired_files.append((file_path.name, txt_file.name))

    random.seed(0)
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * 0.8)
    train_files = paired_files[:split_idx]
    val_files = paired_files[split_idx:]

    move_files(train_files, train_data_path, 'train')
    move_files(val_files, train_data_path, 'val')


def _split_images_labels(images_dir: Path, labels_dir: Path, base_path: Path) -> None:
    """Split files from images_dir/labels_dir into base_path/train and base_path/val."""
    paired = _find_paired_files_in_dir(images_dir, labels_dir)
    random.seed(0)
    random.shuffle(paired)
    split_idx = int(len(paired) * 0.8)

    val_img = base_path / 'val/images'
    val_lbl = base_path / 'val/labels'
    val_img.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    for img_name, lbl_name in paired[split_idx:]:
        shutil.move(str(images_dir / img_name), str(val_img / img_name))
        shutil.move(str(labels_dir / lbl_name), str(val_lbl / lbl_name))

def move_files(files, base_path, data_type):
    base_path = Path(base_path)
    for img_file, txt_file in files:
        # Move image file
        src_img = base_path / img_file
        dst_img = base_path / data_type / 'images' / img_file
        shutil.move(str(src_img), str(dst_img))

        # Move label file
        src_txt = base_path / txt_file
        dst_txt = base_path / data_type / 'labels' / txt_file
        shutil.move(str(src_txt), str(dst_txt))

def create_symlinks(files, base_path, data_type):
    for img_file, txt_file in files:
        src_img_path = os.path.join(base_path, img_file)
        dst_img_path = os.path.join(base_path, data_type, 'images', img_file)
        os.symlink(src_img_path, dst_img_path)

        src_txt_path = os.path.join(base_path, txt_file)
        dst_txt_path = os.path.join(base_path, data_type, 'labels', txt_file)
        os.symlink(src_txt_path, dst_txt_path)

def clean_up(train_data_path):
    for path in ['train', 'val']:
        shutil.rmtree(os.path.join(train_data_path, path), ignore_errors=True)

def copy_and_remove_latest_run_files(model_save_path, project_name, task='detect'):
    """Copy training artefacts from the runs directory to model_save_path."""
    model_save_path = Path(model_save_path)

    # Search the expected task directory first, then fall back to all known tasks
    list_of_dirs: list[Path] = []
    all_tasks = (task, 'detect', 'segment', 'classify', 'pose', 'obb', 'train')
    for candidate_task in all_tasks:
        candidate_base = Path('runs') / candidate_task
        if candidate_base.exists():
            list_of_dirs = list(candidate_base.glob(project_name))
            if list_of_dirs:
                break

    if not list_of_dirs:
        print(f"No 'runs/{task}/{project_name}' directories found. Skipping copy and removal.")
        return

    latest_dir = max(list_of_dirs, key=lambda p: p.stat().st_mtime)

    if latest_dir.exists():
        for item in latest_dir.iterdir():
            dest = model_save_path / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest))

    runs_dir = Path('runs')
    if runs_dir.exists() and runs_dir.is_dir():
        shutil.rmtree(str(runs_dir))

def _find_split_images_dir(root: Path, aliases: list) -> Path | None:
    """Return the first *root/alias/images* directory that exists and has files."""
    for alias in aliases:
        p = root / alias / 'images'
        if p.is_dir() and any(p.iterdir()):
            return p
    return None


def create_yaml(project_name, train_data_path, class_names, save_directory):
    root = Path(train_data_path)

    _TRAIN_ALIASES = ['train', 'Train', 'TRAIN']
    _VAL_ALIASES   = ['val', 'Val', 'VAL', 'valid', 'Valid', 'VALID',
                      'validation', 'Validation', 'VALIDATION']

    train_img = _find_split_images_dir(root, _TRAIN_ALIASES)
    val_img   = _find_split_images_dir(root, _VAL_ALIASES)

    if train_img is not None and val_img is not None:
        # Both splits already exist – use them directly without moving any files.
        train_path = str(train_img).replace('\\', '/')
        val_path   = str(val_img).replace('\\', '/')
    elif train_img is not None and val_img is None:
        # Train split exists but no val split – create a val split from train.
        prepare_data(train_data_path)
        train_path = str(root / 'train').replace('\\', '/')
        val_path   = str(root / 'val').replace('\\', '/')
    else:
        # No recognisable split layout – fall back to the full prepare_data flow
        # (handles flat images/ + labels/ and co-located image/txt pairs).
        prepare_data(train_data_path)
        train_path = str(root / 'train').replace('\\', '/')
        val_path   = str(root / 'val').replace('\\', '/')

    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: [{', '.join(f"'{name}'" for name in class_names)}]
"""
    print(f"Project Name: {project_name}")
    yaml_path = str(Path(save_directory) / f'{project_name}.yaml')
    print(f"YAML Path: {yaml_path}")
    
    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(yaml_content)
    return yaml_path

def _detect_task(model_type: str, custom_model_path: str = None) -> str:
    """Determine the Ultralytics task from the model file name / type string."""
    stem = ""
    if custom_model_path and os.path.isfile(custom_model_path):
        stem = Path(custom_model_path).stem.lower()
    elif model_type:
        stem = model_type.lower()
    if "-seg" in stem:
        return "segment"
    if "-cls" in stem:
        return "classify"
    if "-pose" in stem:
        return "pose"
    if "-obb" in stem:
        return "obb"
    return "detect"


def train_yolo(data_yaml, model_type, img_size, batch, epochs, model_save_path,
               project_name, custom_model_path=None):
    """Train a YOLO model.

    Parameters
    ----------
    model_type:        Ultralytics model name without the .pt suffix (e.g. 'yolov8n').
                       Ignored when custom_model_path is provided.
    custom_model_path: Optional path to a .pt file to use as the training base.
                       When provided this takes precedence over model_type.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resolve which weights file to load.
    # Prefer a locally cached copy in the models/ directory.
    if custom_model_path and os.path.isfile(custom_model_path):
        model_file = custom_model_path
        print(f"Using custom base model: {custom_model_path}")
    elif model_type:
        cached = _MODELS_DIR / f"{model_type}.pt"
        model_file = str(cached) if cached.exists() else f"{model_type}.pt"
    else:
        raise ValueError("Either model_type or a valid custom_model_path must be provided.")

    model = YOLO(model_file).to(device)

    task = _detect_task(model_type, custom_model_path)

    results = model.train(
        data=data_yaml, epochs=epochs, batch=batch,
        imgsz=img_size, name=project_name, save=True,
    )
    copy_and_remove_latest_run_files(model_save_path, project_name, task)
    clean_up(os.path.dirname(data_yaml))
    return results

def parse_args():
    project_name      = sys.argv[1]
    train_data_path   = sys.argv[2]
    class_names       = sys.argv[3].split(',')
    model_save_path   = sys.argv[4]
    model_type        = sys.argv[5]
    img_size          = int(sys.argv[6])
    epochs            = int(sys.argv[7])
    yaml_path         = sys.argv[8]
    batch_size        = int(sys.argv[9])
    # argv[10] is the custom model path (may be an empty string)
    custom_model_path = sys.argv[10] if len(sys.argv) > 10 else None
    if custom_model_path == "":
        custom_model_path = None

    results = train_yolo(
        yaml_path, model_type, img_size, batch_size, epochs,
        model_save_path, project_name, custom_model_path,
    )
    print(f"Training completed. Model saved to {model_save_path}")

if __name__ == '__main__':
    parse_args()