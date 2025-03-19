from pathlib import Path
from shutil import rmtree
from config import logs_base_path

def clear_folder(path):
    for path in Path(path).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)

if __name__ == "__main__":
    print("Running clear Logs folders.")
    clear_folder(logs_base_path)