from pathlib import Path
import os
import cv2

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
    dir = os.path.join(output_dir, f'cam{camid}',f'p{pid}')
    os.makedirs(dir, exist_ok=True)
    remove_oldest_file(dir, '*.jpg',rotate_size)
    path = os.path.join(dir, f'f{frameid}.jpg')
    cv2.imwrite(path, image)