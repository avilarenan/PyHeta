from pathlib import Path
from shutil import rmtree
from config import experiments_base_path, experiment_starting_name_folders
from config import logs_base_path
import os
from glob import glob

def clear_folder(path):
    for path in Path(path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)

def clean_folders_starting_with_path(path):
    path = Path(path)
    pattern = os.path.join(path.parent, f"{experiment_starting_name_folders}*")
    print(pattern)
    for item in glob(pattern):
        print(item)
        if not os.path.isdir(item):
            continue
        rmtree(item)

if __name__ == "__main__":
    print("Running clear experiments folders.")
    clear_folder(experiments_base_path)
    clean_folders_starting_with_path(experiments_base_path)
    print("Running clear log folders.")
    clear_folder(logs_base_path)