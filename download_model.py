import os
import shutil
import zipfile
import gdown

# Local app storage (works without Render persistent disk)
BASE_DIR = "."
MODEL_DIR = os.path.join(BASE_DIR, "sentiment_model")
ZIP_PATH = os.path.join(BASE_DIR, "sentiment_model.zip")
TEMP_EXTRACT_DIR = os.path.join(BASE_DIR, "_extract_tmp")

DRIVE_FILE_ID = "1WxIA4jhBU987_uMXi8ri6mAQL2AX_nEC"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"


def model_ready(model_dir):
    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]
    return os.path.isdir(model_dir) and all(
        os.path.exists(os.path.join(model_dir, f)) for f in required_files
    )


def clean_temp():
    if os.path.exists(TEMP_EXTRACT_DIR):
        shutil.rmtree(TEMP_EXTRACT_DIR, ignore_errors=True)
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)


def find_model_dir(root_dir):
    """
    Find the extracted sentiment_model folder, even if the zip created nested folders.
    """
    for current_root, dirs, files in os.walk(root_dir):
        if "config.json" in files and "tokenizer_config.json" in files:
            return current_root
    return None


def download_and_extract():
    os.makedirs(BASE_DIR, exist_ok=True)

    if model_ready(MODEL_DIR):
        print(f"Model already exists at {MODEL_DIR}. Skipping download.")
        return

    clean_temp()
    os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)

    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, ZIP_PATH, quiet=False)

    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError("Model zip was not downloaded.")

    print("Extracting model zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(TEMP_EXTRACT_DIR)

    discovered_model_dir = find_model_dir(TEMP_EXTRACT_DIR)
    if discovered_model_dir is None:
        raise RuntimeError(
            "Could not find model files after extraction. "
            "Make sure the zip contains the saved sentiment_model files."
        )

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR, ignore_errors=True)

    shutil.move(discovered_model_dir, MODEL_DIR)

    clean_temp()

    if not model_ready(MODEL_DIR):
        raise RuntimeError("Model extraction finished, but required files are missing.")

    print(f"Model is ready at: {MODEL_DIR}")


if __name__ == "__main__":
    download_and_extract()