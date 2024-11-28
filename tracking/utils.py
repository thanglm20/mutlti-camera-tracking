from pathlib import Path
import os
import cv2
import numpy as np
from datetime import datetime, timezone

def remove_oldest_file(dir, ext, rotate_size=5):
    files = [f for f in Path(dir).glob(ext) if f.is_file()]
    if len(files) > rotate_size:
        files.sort(key=lambda x: x.stat().st_mtime)
        # Remove the oldest file
        oldest_file = files[0]
        oldest_file.unlink()  # Deletes the file

def save_person_image(output_dir, camid, frameid, pid, image, rotate_size=5):
    if image.size == 0:
        return
    dir = os.path.join(output_dir, f'{camid}',f'p{pid}')
    os.makedirs(dir, exist_ok=True)
    remove_oldest_file(dir, '*.jpg',rotate_size)
    path = os.path.join(dir, f'f{frameid}.jpg')
    cv2.imwrite(path, image)
    return path

def read_image(image_path):
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        return image
    else:
        None

def get_current_utc_iso():
    utc_time = datetime.now(timezone.utc)
    iso_format = utc_time.isoformat()
    return iso_format