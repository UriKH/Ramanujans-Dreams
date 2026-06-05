import shutil
from pathlib import Path
import zipfile


def unzip_file(zip_path, extract_to_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        print(f"Files extracted to: {extract_to_folder}")


def copy_file(source_path, destination_path, move=False):
    """
    Copies a file or an entire directory to a destination.
    """
    src = Path(source_path)
    dest = Path(destination_path)

    try:
        if not src.exists():
            print(f"Error: Source '{src}' does not exist.")
            return

        if src.is_file():
            # Ensure the parent directory of the destination exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
        elif src.is_dir():
            # Copy the entire directory tree
            # dirs_exist_ok=True allows copying into an existing directory
            if move:
                shutil.move(src, dest)
            else:
                shutil.copytree(src, dest, dirs_exist_ok=True)
    except Exception as e:
        print(f"An error occurred: {e}")


def move_all(source_dir, dest_dir):
    src = Path(source_dir)
    dest = Path(dest_dir)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    shutil.rmtree(src)


if __name__ == "__main__":
    TEMP_DIR = 'setup/temp_files/'

    unzip_file('packed.zip', TEMP_DIR)
    try:
        file = Path('packed.zip')
        file.unlink()
    except FileNotFoundError:
        print("packed file already deleted")

    copy_file("setup/copilot-instructions.md", "../.github/", move=True)
    move_all(TEMP_DIR, ".")