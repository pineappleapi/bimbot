#!/usr/bin/env -S python3 -O

"""Upload module for bim-bot Flask app."""

import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "m4v"}


def allowed_file(filename: str) -> bool:
    """Return True if filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rotate_existing_uploads(
    target_dir: Path,
    new_base: str = "new_upload",
    old_base: str = "old_upload",
) -> Optional[Path]:
    """
    If a file named new_upload.* exists in target_dir, rename it to old_upload(n).*
    preserving the extension and choosing n = max existing + 1.


    Returns:
      Path to the rotated video (old_upload(n).ext), or None if nothing was rotated.
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    existing_new = None
    # Find exact match stem == new_base, regardless of extension
    for p in target_dir.iterdir():
        if p.is_file() and p.stem == new_base:
            existing_new = p
            break

    if existing_new is None:
        return None  # nothing to rotate

    # Determine next n for old_upload(n)
    max_n = 0
    pattern = re.compile(rf"^{re.escape(old_base)}\((\d+)\)$")  # match stem like old_upload(3)
    for p in target_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.stem)
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_n:
                        max_n = n
                except ValueError:
                    pass
    next_n = max_n + 1

    # Rename existing video preserving extension
    rotated_video = target_dir / f"{old_base}({next_n}){existing_new.suffix}"
    existing_new.rename(rotated_video)

    return rotated_video


def save_uploaded_file(
    file: FileStorage,
    target_dir: Path,
    *,
    new_base: str = "new_upload",
    old_base: str = "old_upload",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Save an uploaded video to target_dir following the naming policy:
      1) If new_upload.<ext> exists, rotate it to old_upload(n).<ext> 
      2) Save the incoming file as new_upload.<ext> (preserve the new file's extension).

    Args:
      file: Werkzeug FileStorage from Flask (request.files["video"])
      target_dir: target directory as Path
      new_base: base name for the latest upload ("new_upload")
      old_base: base name for rotated files ("old_upload")

    Returns:
      (ok: bool, info: dict)
      info contains keys:
        - "video_path": str
        - "rotated_video_path": Optional[str]
        - "message": str
        - "error": Optional[str]
    """
    info: Dict[str, Any] = {
        "video_path": None,
        "rotated_video_path": None,
        "message": "",
        "error": None,
    }

    if file is None or not getattr(file, "filename", ""):
        info["error"] = "No file provided or empty filename."
        return False, info

    safe_name = secure_filename(file.filename or "")
    if not allowed_file(safe_name):
        info["error"] = f"Unsupported file type. Allowed: {sorted(list(ALLOWED_EXTENSIONS))}"
        return False, info

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract and normalize extension
    _, ext = os.path.splitext(safe_name)
    ext = ext.lower()

    # Rotate existing new_upload.* if present
    rotated = rotate_existing_uploads(
        target_dir=target_dir,
        new_base=new_base,
        old_base=old_base
    )
    if rotated is not None:
        info["rotated_video_path"] = str(rotated)

    # Save incoming file as new_upload.<ext>
    new_upload_path = target_dir / f"{new_base}{ext}"
    try:
        file.save(new_upload_path)
        info["video_path"] = str(new_upload_path)
        info["message"] = "Video saved as new upload."
    except Exception as e:
        info["error"] = f"Failed to save file: {e}"
        return False, info

    return True, info
