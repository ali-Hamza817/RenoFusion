import subprocess
import tempfile
import os

val_dir = "/home/administrator/Desktop/RCC/validation_cases/TCGA-B0-4698"
img_path = os.path.join(val_dir, "ct_scan.nii.gz")
tmp_dir = tempfile.mkdtemp()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "2"
try:
    print(f"Running TotalSegmentator on {img_path} -> {tmp_dir}")
    res = subprocess.run([
        "TotalSegmentator", 
        "-i", img_path, 
        "-o", tmp_dir, 
        "-rs", "kidney_left", "kidney_right", 
        "--fast"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    print("Success. Checking files:")
    print(os.listdir(tmp_dir))
except subprocess.CalledProcessError as e:
    print("Error:", e.stderr.decode())
