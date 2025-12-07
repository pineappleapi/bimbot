#main.py

import subprocess
import sys
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import shutil



def pick_file(title="Select a file", filetypes=(("All files", "*.*"),)):
    root = Tk()
    root.withdraw()
    path = askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path

def uploadfile():
    # Allow images and MP4 videos
    file_path = pick_file(filetypes=(("Media Files", "*.png *.jpg *.jpeg *.mp4"),))
    if not file_path:
        print("No file selected.")
        return
    
    src = Path(file_path)

    # Save to your desired fixed folder
    TARGET_DIR = Path(r"D:\Sophia\Videos\bimbot_loaded_videos")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)  # create if missing

    dest = TARGET_DIR / src.name
    shutil.copy2(src, dest)  # Copies the file (overwrites if same name exists)
    print(f"File saved to: {dest}")



def run_depth_anything(args=None):
    project_root = Path(__file__).parent
    # IMPORTANT: point to the metric_depth run.py
    run_py = project_root / "DepthAnything" / "metric_depth" / "run.py"

    # Example paths — replace if needed
    img_path = project_root / "DepthAnything" / "min3d" / "Underground" / "und_1_rgb"
    outdir = project_root / "depth_images"

    cmd = [
        sys.executable,
        str(run_py),
        "--encoder", "vitl",
        "--img-path", str(img_path),
        "--outdir", str(outdir),
        "--max-depth", "20",
        # Enable these flags:
        "--save-numpy",
        "--pred-only",
        # (optional) use your preferred checkpoint:
        "--load-from", r"D:\\BIM-BOT\\DepthAnything\\metric_depth\\checkpoints\\depth_anything_v2_metric_vkitti_vitl.pth",
    ]

    if args:
        cmd.extend(args)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] run.py exited with code {e.returncode}")

if __name__ == "__main__":
    #run_depth_anything()
    print("Depth Anything processing completed.")
    uploadfile()  # keep your existing call if needed
