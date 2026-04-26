"""dataset.py – Utilities for importing datasets, especially Roboflow ZIPs.

Supports Roboflow YOLOv8 export format:
  data.yaml (may appear multiple times in the archive)
  train/images/*.jpg   train/labels/*.txt
  valid/images/*.jpg   valid/labels/*.txt
  test/images/*.jpg    test/labels/*.txt   (optional)

The data.yaml 'names' field may be either a plain list (standard Ultralytics)
or an integer-keyed dict (Roboflow export).  Both are normalised to a list.
"""

import shutil
import warnings
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
#  YAML helpers  (PyYAML is a transitive dep of ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(yaml_path: Path) -> dict:
    import yaml
    with open(str(yaml_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(yaml_path: Path, data: dict) -> None:
    import yaml
    with open(str(yaml_path), "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Name normalisation
# ─────────────────────────────────────────────────────────────────────────────

def parse_names(names_data) -> list:
    """Convert Roboflow dict-format or standard list-format names to a list.

    Roboflow exports:
        names:
          0: cat
          1: dog
    PyYAML loads this as {0: 'cat', 1: 'dog'}.  Standard Ultralytics uses a
    plain list.  Both forms are accepted and returned as a plain list.
    """
    if isinstance(names_data, list):
        return [str(n) for n in names_data]
    if isinstance(names_data, dict):
        return [str(names_data[k]) for k in sorted(names_data.keys())]
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Roboflow ZIP extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_roboflow_zip(zip_path: str, extract_dir: str, progress_callback=None) -> tuple:
    """Extract a Roboflow YOLOv8-format ZIP archive.

    Parameters
    ----------
    zip_path   : path to the .zip file to import
    extract_dir: parent directory; a sub-folder named after the zip stem is
                 created inside it (avoids collisions on re-import)
    progress_callback : optional callable(current: int, total: int, message: str)
                 called periodically during extraction to report progress.

    Returns
    -------
    (dataset_root, yaml_path, class_names)
      dataset_root – absolute path to the extracted folder
      yaml_path    – absolute path to the patched data.yaml
      class_names  – ordered list of class name strings
    """
    zip_path = Path(zip_path).resolve()
    extract_dir = Path(extract_dir).resolve()
    dataset_root = extract_dir / zip_path.stem
    dataset_root.mkdir(parents=True, exist_ok=True)

    _extract_zip(zip_path, dataset_root, progress_callback=progress_callback)

    yaml_path = _find_yaml(dataset_root)
    if yaml_path is None:
        raise FileNotFoundError(
            f"No data.yaml found after extracting '{zip_path.name}'. "
            "Please confirm this is a Roboflow YOLOv8 export."
        )

    data = _load_yaml(yaml_path)
    class_names = parse_names(data.get("names", []))

    # Rewrite YAML: absolute paths, normalised names, 'val' alias
    _patch_yaml(yaml_path, dataset_root)

    return str(dataset_root), str(yaml_path), class_names


def _extract_zip(zip_path: Path, dest: Path, progress_callback=None) -> None:
    """Extract the ZIP, writing the first occurrence of data.yaml only.

    Roboflow historically embeds data.yaml under multiple sub-paths in the
    same archive, which causes name-collision warnings on extraction.
    We deduplicate by only writing the first data.yaml encountered.

    progress_callback(current: int, total: int, message: str) is called
    periodically so the caller can update a progress indicator.
    """
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        members = zf.namelist()
        total = len(members)
        yaml_written = False
        for i, member in enumerate(members):
            is_yaml = Path(member).name.lower() in ("data.yaml", "data.yml")
            if is_yaml:
                if yaml_written:
                    continue
                yaml_written = True

            out_path = dest / member
            if member.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(str(out_path), "wb") as dst:
                    shutil.copyfileobj(src, dst)

            if progress_callback and (i % 20 == 0 or i == total - 1):
                progress_callback(i + 1, total, Path(member).name)


def _find_yaml(dataset_root: Path):
    """Return the top-level data.yaml, or the first yaml found recursively."""
    direct = dataset_root / "data.yaml"
    if direct.exists():
        return direct
    candidates = sorted(dataset_root.rglob("data.yaml"))
    return candidates[0] if candidates else None


def _patch_yaml(yaml_path: Path, dataset_root: Path) -> None:
    """Rewrite data.yaml so Ultralytics can find the dataset from any CWD.

    Changes made:
    * path keys (train / val / valid / test) → absolute paths
    * names dict → plain list
    * 'valid' key aliased to 'val' (Roboflow uses 'valid', Ultralytics wants 'val')
    * top-level 'path' key set to dataset_root
    * val/valid paths that cannot be resolved fall back to the train split
    * test path that cannot be resolved is removed
    """
    data = _load_yaml(yaml_path)

    # Normalise names to plain list
    if isinstance(data.get("names"), dict):
        data["names"] = parse_names(data["names"])
        data["nc"] = len(data["names"])

    yaml_dir = yaml_path.parent

    def _resolve(value: str) -> str | None:
        """Return an absolute path string if *value* resolves to an existing path, else None."""
        p = Path(value)
        if p.is_absolute():
            return str(p).replace("\\", "/") if p.exists() else None
        # Try relative to dataset_root first, then to the yaml's own directory
        for base in (dataset_root, yaml_dir):
            candidate = (base / value).resolve()
            if candidate.exists():
                return str(candidate).replace("\\", "/")
        return None

    # Resolve relative path values to absolute
    for key in ("train", "val", "valid", "test"):
        if key not in data:
            continue
        resolved = _resolve(str(data[key]))
        if resolved is not None:
            data[key] = resolved

    # 'valid' → 'val' alias for Ultralytics
    if "valid" in data and "val" not in data:
        data["val"] = data["valid"]

    # Top-level path key (Ultralytics uses this as dataset root)
    data["path"] = str(dataset_root).replace("\\", "/")

    # Determine resolved train path to use as a fallback for missing splits
    train_resolved = _resolve(str(data["train"])) if data.get("train") else None

    # val/valid: if the path still does not exist on disk, fall back to train
    for key in ("val", "valid"):
        if key not in data:
            continue
        current = str(data[key])
        if not Path(current).exists():
            if train_resolved:
                warnings.warn(
                    f"'{key}' path '{current}' not found. "
                    "Falling back to the train split for validation.",
                    UserWarning,
                    stacklevel=2,
                )
                data[key] = train_resolved
            else:
                warnings.warn(
                    f"'{key}' path '{current}' not found and no train fallback is available. "
                    f"Removing '{key}' from the dataset configuration.",
                    UserWarning,
                    stacklevel=2,
                )
                del data[key]

    # test: remove entirely if the path does not exist
    if "test" in data:
        current = str(data["test"])
        if not Path(current).exists():
            warnings.warn(
                f"'test' path '{current}' not found. "
                "Removing 'test' from the dataset configuration.",
                UserWarning,
                stacklevel=2,
            )
            del data["test"]

    _write_yaml(yaml_path, data)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset statistics
# ─────────────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def count_dataset_images(dataset_root: str) -> dict:
    """Count images in each split found under dataset_root.

    Returns a dict like {'train': 800, 'valid': 100, 'test': 50}.
    """
    root = Path(dataset_root)
    counts = {}
    for split in ("train", "valid", "val", "test"):
        img_dir = root / split / "images"
        if img_dir.exists():
            n = sum(
                1 for f in img_dir.iterdir()
                if f.suffix.lower() in _IMG_EXTS
            )
            if n:
                counts[split] = n
    return counts
