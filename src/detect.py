import shutil
import os
import cv2
import mimetypes
from pathlib import Path
from ultralytics import YOLO
from typing import List, Union, Callable, Optional
from datetime import datetime

VALID_IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.ppm',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.PPM'
}

VALID_VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',
    '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV'
}

def is_valid_image(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid image by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_IMAGE_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('image/'):
            return False
            
        return True
    except Exception:
        return False

def is_valid_video(file_path: Union[str, Path]) -> bool:
    """Check if a file is a valid video by examining both extension and mime type"""
    try:
        file_path = Path(file_path)
        # Check file extension
        if file_path.suffix.lower() not in {ext.lower() for ext in VALID_VIDEO_EXTENSIONS}:
            return False
            
        # Check mime type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type or not mime_type.startswith('video/'):
            return False
            
        return True
    except Exception:
        return False

def normalize_path(file_path: Union[str, Path]) -> Path:
    """Normalize path to handle Japanese characters and different path formats"""
    try:
        return Path(file_path).resolve()
    except Exception:
        raise ValueError(f"Invalid path: {file_path}")

def get_media_files(directory: Union[str, Path]) -> tuple[List[Path], List[Path]]:
    """Recursively find all valid images and videos in a directory"""
    directory = normalize_path(directory)
    image_files = []
    video_files = []
    
    try:
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            if is_valid_image(file_path):
                image_files.append(file_path)
            elif is_valid_video(file_path):
                video_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
        return [], []
        
    return sorted(image_files), sorted(video_files)

def process_video(
    video_path: Path,
    model,
    output_dir: Path,
    conf_threshold: float = 0.5,
    progress_callback: Optional[Callable] = None,
    cancel_flag: Optional[Callable] = None,
    half: bool = False,
):
    """Process a video file and save detection results.

    progress_callback(frac: float, msg: str) is called every 30 frames where
    frac is in [0, 1] representing progress through this video.
    cancel_flag() returns True if processing should be stopped.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    video_name = video_path.stem
    video_output_dir = output_dir / f"{timestamp}_{video_name}"
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Create video writer
    output_video_path = video_output_dir / f"{video_name}_detection.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_count = 0
    detection_count = 0

    while True:
        if cancel_flag and cancel_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Update progress every 30 frames
        if frame_count % 30 == 0:
            frac = frame_count / total_frames
            if progress_callback:
                progress_callback(
                    frac,
                    f"{video_path.name}: frame {frame_count}/{total_frames}",
                )
            else:
                print(
                    f"\rProcessing video: {frac * 100:.1f}% ({frame_count}/{total_frames} frames)",
                    end="",
                )

        # Detect objects in frame
        results = model.predict(frame, save=False, conf=conf_threshold, half=half, verbose=False)

        # Only process frames with detections above threshold
        if len(results[0].boxes) > 0:
            annotated_frame = results[0].plot()

            frame_path = video_output_dir / f"frame_{detection_count:04d}.jpg"
            cv2.imwrite(str(frame_path), annotated_frame)

            txt_path = video_output_dir / f"frame_{detection_count:04d}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for box in results[0].boxes:
                    if box.conf[0] >= conf_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = model.names[int(box.cls[0])]
                        confidence = box.conf[0]
                        f.write(f"{label} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

            detection_count += 1
            out.write(annotated_frame)
        else:
            out.write(frame)

        frame_count += 1

    # Clean up
    cap.release()
    out.release()

    print(f"\nVideo processing complete. {detection_count} frames with detections saved.")
    print(f"Output video saved to: {output_video_path}")

    return video_output_dir


def get_model_info(model_path: str) -> dict:
    """Return basic metadata about a YOLO .pt file without full loading."""
    p = Path(model_path)
    info = {
        "name": p.name,
        "size_mb": p.stat().st_size / 1_048_576 if p.exists() else 0,
        "num_classes": None,
        "class_names": [],
        "task": "unknown",
    }
    try:
        model = YOLO(model_path)
        info["num_classes"] = len(model.names)
        info["class_names"] = list(model.names.values())
        info["task"] = getattr(model, "task", "detect") or "detect"
    except Exception:
        pass
    return info

def move_detection_results(source_dir, target_dir):
    source_dir = normalize_path(source_dir)
    target_dir = normalize_path(target_dir)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in source_dir.iterdir():
        target_file = target_dir / file_path.name
        
        # Remove existing file/directory if it exists
        if target_file.exists():
            if target_file.is_dir():
                shutil.rmtree(str(target_file))
            else:
                target_file.unlink()
        
        # Move the file
        if file_path.is_file():
            shutil.move(str(file_path), str(target_file))
        else:
            shutil.move(str(file_path), str(target_dir))
    
    # Clean up source directory
    shutil.rmtree(str(source_dir))

def _find_latest_predict_run():
    """Find the most recently modified prediction run directory under runs/."""
    candidates = []
    runs_base = Path('runs')
    if not runs_base.exists():
        return None
    for task_dir in runs_base.iterdir():
        if task_dir.is_dir():
            for run_dir in task_dir.iterdir():
                if run_dir.is_dir():
                    candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def detect_images(
    images_folder,
    model_path,
    callback=None,
    progress_callback: Optional[Callable] = None,
    image_result_callback: Optional[Callable] = None,
    conf_threshold: float = 0.5,
    half: bool = False,
    workers: int = 4,
    cancel_flag: Optional[Callable] = None,
    task: Optional[str] = None,
):
    """Run YOLO detection on all images/videos in a folder.

    Parameters
    ----------
    progress_callback(current: int, total: int, msg: str)
        Called after each image or video with the running count.
    image_result_callback(image_path: str)
        Called after each image is processed and saved, with the path of the
        result image so the GUI can display it incrementally.
    cancel_flag()
        Callable that returns True when the user wants to abort.
    task
        Explicit YOLO task type ('detect', 'segment', 'classify', 'pose', 'obb').
        Required for exported formats such as TensorRT (.engine) or ONNX (.onnx)
        that do not embed task metadata.  When None the task is inferred
        automatically (works for .pt files but may fail for exported formats).
    """
    model = YOLO(model_path, task=task)

    images_folder = normalize_path(images_folder)
    image_files, video_files = get_media_files(images_folder)

    if not image_files and not video_files:
        print("No valid media files found in the directory")
        if callback:
            # Create the results dir so _on_detection_complete can scan it
            empty_results = Path(images_folder) / 'results'
            empty_results.mkdir(parents=True, exist_ok=True)
            callback(str(empty_results))
        return

    total = len(image_files) + len(video_files)
    current = 0

    results_dir = Path(images_folder) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Process images one-by-one so we can report per-image progress
    for i, img_path in enumerate(image_files):
        if cancel_flag and cancel_flag():
            break
        if progress_callback:
            progress_callback(current, total, f"Image {i + 1}/{len(image_files)}: {img_path.name}")
        model.predict(
            str(img_path), save=True, save_txt=True, imgsz=640,
            conf=conf_threshold, half=half, workers=workers, verbose=False,
        )
        latest_run_dir = _find_latest_predict_run()
        if latest_run_dir:
            move_detection_results(latest_run_dir, results_dir)
        # Notify caller about the newly saved result image so the GUI can
        # display it immediately rather than waiting for all images to finish.
        if image_result_callback:
            # Fast path: YOLO normally preserves the original filename.
            result_candidate = results_dir / img_path.name
            if not (result_candidate.is_file() and is_valid_image(result_candidate)):
                # Fallback: scan for any file whose stem matches (handles
                # cases where YOLO converts the extension, e.g. PNG → JPEG).
                result_candidate = None
                for result_file in results_dir.iterdir():
                    if (result_file.is_file()
                            and result_file.stem == img_path.stem
                            and is_valid_image(result_file)):
                        result_candidate = result_file
                        break
            if result_candidate:
                image_result_callback(str(result_candidate))
        current += 1

    # Process videos
    for j, video_file in enumerate(video_files):
        if cancel_flag and cancel_flag():
            break

        def _vid_cb(frac: float, msg: str, _cur=current, _tot=total):
            if progress_callback:
                # Map video frame fraction into the global progress slot for this video
                progress_callback(int(_cur + frac), _tot, msg)

        if progress_callback:
            progress_callback(current, total, f"Video {j + 1}/{len(video_files)}: {video_file.name}")

        process_video(
            video_file, model, results_dir,
            conf_threshold=conf_threshold,
            progress_callback=_vid_cb,
            cancel_flag=cancel_flag,
            half=half,
        )
        current += 1
        if progress_callback:
            progress_callback(current, total, f"Video {j + 1}/{len(video_files)} complete")

    if progress_callback:
        progress_callback(total, total, "Detection complete")

    if callback:
        callback(str(results_dir))