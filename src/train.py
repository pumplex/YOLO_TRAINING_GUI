import os
import sys
import torch
import shutil
import glob
import random
import mimetypes
from pathlib import Path
from ultralytics import YOLO
import json

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

def copy_and_remove_latest_run_files(model_save_path, project_name, task='detect', source_dir=None):
    """Copy training artefacts from the runs directory to model_save_path.

    Parameters
    ----------
    model_save_path : str or Path
        Destination directory where artefacts should be copied.
    project_name : str
        The training run name (used only for the fallback directory search).
    task : str
        YOLO task type used for the fallback directory search.
    source_dir : str or Path, optional
        Exact path to the run directory produced by YOLO (i.e.
        ``model.trainer.save_dir``).  When provided the directory search is
        skipped entirely, which avoids mismatches caused by YOLO appending a
        deduplication suffix to the project name (e.g. ``run2``, ``run3``).
    """
    model_save_path = Path(model_save_path)

    # ── Resolve the source directory ─────────────────────────────────────────
    if source_dir is not None:
        latest_dir = Path(source_dir)
        if not latest_dir.exists():
            print(f"Source run directory '{latest_dir}' does not exist. Skipping copy.")
            return
    else:
        # Fallback: search the runs tree relative to CWD for a matching name.
        list_of_dirs: list[Path] = []
        all_tasks = (task, 'detect', 'segment', 'classify', 'pose', 'obb', 'train')
        for candidate_task in all_tasks:
            candidate_base = Path('runs') / candidate_task
            if candidate_base.exists():
                list_of_dirs = list(candidate_base.glob(project_name))
                if list_of_dirs:
                    break

        if not list_of_dirs:
            print(f"No 'runs/{task}/{project_name}' directories found. Skipping copy.")
            return

        latest_dir = max(list_of_dirs, key=lambda p: p.stat().st_mtime)

    # ── Copy artefacts ────────────────────────────────────────────────────────
    model_save_path.mkdir(parents=True, exist_ok=True)
    copy_ok = True
    try:
        for item in latest_dir.iterdir():
            dest = model_save_path / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest))
        print(f"Training artefacts copied from '{latest_dir}' to '{model_save_path}'.")
    except Exception as exc:
        print(f"Error copying training artefacts: {exc}")
        copy_ok = False

    if not copy_ok:
        print(
            f"Copy did not complete cleanly. "
            f"The source run directory '{latest_dir}' has NOT been removed."
        )
        return

    # ── Verify every copied item then delete the source run folder ────────────
    # Only the specific run directory that was just copied is removed; the rest
    # of the runs tree (sibling runs, parent task folder, etc.) is untouched.
    verify_ok = True
    for item in latest_dir.iterdir():
        dest = model_save_path / item.name
        if not dest.exists():
            print(f"Verification failed: '{dest}' not found in destination.")
            verify_ok = False
            break
        if item.is_file():
            if item.stat().st_size != dest.stat().st_size:
                print(
                    f"Verification failed: size mismatch for '{item.name}' "
                    f"(source {item.stat().st_size} B, dest {dest.stat().st_size} B)."
                )
                verify_ok = False
                break

    if verify_ok:
        try:
            shutil.rmtree(str(latest_dir))
            print(f"Source run directory '{latest_dir}' removed after successful copy.")
        except Exception as exc:
            print(f"Could not remove source run directory '{latest_dir}': {exc}")
    else:
        print(
            f"Verification did not pass. "
            f"The source run directory '{latest_dir}' has NOT been removed."
        )

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
               project_name, custom_model_path=None, extra_params=None):
    """Train a YOLO model.

    Parameters
    ----------
    model_type:        Ultralytics model name without the .pt suffix (e.g. 'yolov8n').
                       Ignored when custom_model_path is provided.
    custom_model_path: Optional path to a .pt file to use as the training base.
                       When provided this takes precedence over model_type.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ep = extra_params or {}

    # Resolve which weights file to load.
    # When resuming, the checkpoint (last.pt) must be the model file so that
    # Ultralytics can restore optimizer state and epoch count.
    resume_ckpt = ep.get('resume_checkpoint', '')
    if resume_ckpt and os.path.isfile(resume_ckpt):
        model_file = resume_ckpt
        print(f"Resuming from checkpoint: {resume_ckpt}")
    elif custom_model_path and os.path.isfile(custom_model_path):
        model_file = custom_model_path
        print(f"Using custom base model: {custom_model_path}")
    elif model_type:
        cached = _MODELS_DIR / f"{model_type}.pt"
        model_file = str(cached) if cached.exists() else f"{model_type}.pt"
    else:
        raise ValueError("Either model_type or a valid custom_model_path must be provided.")

    model = YOLO(model_file).to(device)

    task = _detect_task(model_type, custom_model_path)

    train_kwargs = dict(
        data=data_yaml, epochs=epochs, batch=batch,
        imgsz=img_size, name=project_name,
    )

    # workers
    w = ep.get('workers')
    if w is not None:
        train_kwargs['workers'] = int(w)

    # save (default True)
    train_kwargs['save'] = bool(ep.get('save', True))

    # time (0 = disabled / not passed)
    t = ep.get('time', 0)
    try:
        if t and float(t) > 0:
            train_kwargs['time'] = float(t)
    except (ValueError, TypeError):
        pass

    # patience
    if 'patience' in ep:
        try:
            train_kwargs['patience'] = int(ep['patience'])
        except (ValueError, TypeError):
            pass

    # save_period (-1 = disabled by default)
    if 'save_period' in ep:
        try:
            train_kwargs['save_period'] = int(ep['save_period'])
        except (ValueError, TypeError):
            pass

    # cache
    if 'cache' in ep:
        train_kwargs['cache'] = bool(ep['cache'])

    # resume
    if ep.get('resume', False):
        train_kwargs['resume'] = True

    # freeze (0 or negative = disabled)
    try:
        freeze = int(ep.get('freeze', 0))
        if freeze > 0:
            train_kwargs['freeze'] = freeze
    except (ValueError, TypeError):
        pass

    # lr0
    if 'lr0' in ep:
        try:
            train_kwargs['lr0'] = float(ep['lr0'])
        except (ValueError, TypeError):
            pass

    # lrf
    if 'lrf' in ep:
        try:
            train_kwargs['lrf'] = float(ep['lrf'])
        except (ValueError, TypeError):
            pass

    # momentum
    if 'momentum' in ep:
        try:
            train_kwargs['momentum'] = float(ep['momentum'])
        except (ValueError, TypeError):
            pass

    # weight_decay
    if 'weight_decay' in ep:
        try:
            train_kwargs['weight_decay'] = float(ep['weight_decay'])
        except (ValueError, TypeError):
            pass

    # optimizer
    opt = ep.get('optimizer', '')
    if opt:
        train_kwargs['optimizer'] = opt

    # val (default True)
    if 'val' in ep:
        train_kwargs['val'] = bool(ep['val'])

    # max_det (only when val is True)
    if ep.get('val', True) and 'max_det' in ep:
        try:
            train_kwargs['max_det'] = int(ep['max_det'])
        except (ValueError, TypeError):
            pass

    # device (blank / 'auto' → omit so Ultralytics auto-selects)
    device_val = ep.get('device', 'auto')
    if device_val and str(device_val).strip().lower() not in ('', 'auto'):
        train_kwargs['device'] = str(device_val).strip()

    # seed
    if 'seed' in ep:
        try:
            train_kwargs['seed'] = int(ep['seed'])
        except (ValueError, TypeError):
            pass

    # deterministic (default True)
    if 'deterministic' in ep:
        train_kwargs['deterministic'] = bool(ep['deterministic'])

    # verbose (default True)
    if 'verbose' in ep:
        train_kwargs['verbose'] = bool(ep['verbose'])

    # exist_ok (default False)
    if 'exist_ok' in ep:
        train_kwargs['exist_ok'] = bool(ep['exist_ok'])

    # single_cls (default False)
    if 'single_cls' in ep:
        train_kwargs['single_cls'] = bool(ep['single_cls'])

    # classes – comma-separated IDs string → list of ints (blank = omit)
    classes_raw = ep.get('classes', '')
    if classes_raw and str(classes_raw).strip():
        try:
            train_kwargs['classes'] = [int(x.strip()) for x in str(classes_raw).split(',') if x.strip()]
        except (ValueError, TypeError):
            pass

    # fraction (default 1.0)
    if 'fraction' in ep:
        try:
            frac = float(ep['fraction'])
            if 0.0 < frac <= 1.0:
                train_kwargs['fraction'] = frac
        except (ValueError, TypeError):
            pass

    # profile (default False)
    if 'profile' in ep:
        train_kwargs['profile'] = bool(ep['profile'])

    # amp (default True)
    if 'amp' in ep:
        train_kwargs['amp'] = bool(ep['amp'])

    # rect (default False)
    if 'rect' in ep:
        train_kwargs['rect'] = bool(ep['rect'])

    # multi_scale (0.0 = disabled)
    if 'multi_scale' in ep:
        try:
            ms = float(ep['multi_scale'])
            if ms > 0:
                train_kwargs['multi_scale'] = ms
        except (ValueError, TypeError):
            pass

    # cos_lr (default False)
    if 'cos_lr' in ep:
        train_kwargs['cos_lr'] = bool(ep['cos_lr'])

    # close_mosaic (default 10)
    if 'close_mosaic' in ep:
        try:
            train_kwargs['close_mosaic'] = int(ep['close_mosaic'])
        except (ValueError, TypeError):
            pass

    # plots (default True)
    if 'plots' in ep:
        train_kwargs['plots'] = bool(ep['plots'])

    # compile (default False; "False" string → False bool)
    compile_val = ep.get('compile', 'False')
    if str(compile_val).lower() in ('false', '0', ''):
        train_kwargs['compile'] = False
    elif str(compile_val).lower() in ('true', '1'):
        train_kwargs['compile'] = True
    else:
        train_kwargs['compile'] = str(compile_val)

    # warmup_epochs (default 3.0)
    if 'warmup_epochs' in ep:
        try:
            train_kwargs['warmup_epochs'] = float(ep['warmup_epochs'])
        except (ValueError, TypeError):
            pass

    # warmup_momentum (default 0.8)
    if 'warmup_momentum' in ep:
        try:
            train_kwargs['warmup_momentum'] = float(ep['warmup_momentum'])
        except (ValueError, TypeError):
            pass

    # warmup_bias_lr (default 0.1)
    if 'warmup_bias_lr' in ep:
        try:
            train_kwargs['warmup_bias_lr'] = float(ep['warmup_bias_lr'])
        except (ValueError, TypeError):
            pass

    # box loss weight (default 7.5)
    if 'box' in ep:
        try:
            train_kwargs['box'] = float(ep['box'])
        except (ValueError, TypeError):
            pass

    # cls loss weight (default 0.5) – stored as 'cls_loss' in GUI to avoid name clash
    if 'cls_loss' in ep:
        try:
            train_kwargs['cls'] = float(ep['cls_loss'])
        except (ValueError, TypeError):
            pass

    # cls_pw (default 0.0)
    if 'cls_pw' in ep:
        try:
            train_kwargs['cls_pw'] = float(ep['cls_pw'])
        except (ValueError, TypeError):
            pass

    # dfl loss weight (default 1.5)
    if 'dfl' in ep:
        try:
            train_kwargs['dfl'] = float(ep['dfl'])
        except (ValueError, TypeError):
            pass

    # nbs (default 64)
    if 'nbs' in ep:
        try:
            train_kwargs['nbs'] = int(ep['nbs'])
        except (ValueError, TypeError):
            pass

    # pose loss weight (default 12.0)
    if 'pose' in ep:
        try:
            train_kwargs['pose'] = float(ep['pose'])
        except (ValueError, TypeError):
            pass

    # kobj loss weight (default 1.0)
    if 'kobj' in ep:
        try:
            train_kwargs['kobj'] = float(ep['kobj'])
        except (ValueError, TypeError):
            pass

    # rle loss weight (default 1.0)
    if 'rle' in ep:
        try:
            train_kwargs['rle'] = float(ep['rle'])
        except (ValueError, TypeError):
            pass

    # angle loss weight (default 1.0)
    if 'angle' in ep:
        try:
            train_kwargs['angle'] = float(ep['angle'])
        except (ValueError, TypeError):
            pass

    # overlap_mask (default True)
    if 'overlap_mask' in ep:
        train_kwargs['overlap_mask'] = bool(ep['overlap_mask'])

    # mask_ratio (default 4)
    if 'mask_ratio' in ep:
        try:
            train_kwargs['mask_ratio'] = int(ep['mask_ratio'])
        except (ValueError, TypeError):
            pass

    # dropout (default 0.0)
    if 'dropout' in ep:
        try:
            train_kwargs['dropout'] = float(ep['dropout'])
        except (ValueError, TypeError):
            pass

    results = model.train(**train_kwargs)
    # Use the exact save directory recorded by the trainer so the copy always
    # targets the right folder, even when YOLO appends a deduplication suffix.
    trainer_save_dir = getattr(getattr(model, 'trainer', None), 'save_dir', None)
    copy_and_remove_latest_run_files(model_save_path, project_name, task,
                                     source_dir=trainer_save_dir)
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
    custom_model_path = sys.argv[10] if len(sys.argv) > 10 else None
    if custom_model_path == "":
        custom_model_path = None

    extra_json = sys.argv[11] if len(sys.argv) > 11 else '{}'
    try:
        extra_params = json.loads(extra_json) if extra_json else {}
    except Exception:
        extra_params = {}

    results = train_yolo(
        yaml_path, model_type, img_size, batch_size, epochs,
        model_save_path, project_name, custom_model_path, extra_params,
    )
    print(f"Training completed. Model saved to {model_save_path}")

if __name__ == '__main__':
    parse_args()