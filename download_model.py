import os
import zipfile
import gdown

MODEL_DIR = "sentiment_model"
ZIP_FILE = "sentiment_model.zip"

# Replace this with your Drive file ID
DRIVE_FILE_ID = "PUT_YOUR_FILE_ID_HERE"

URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"


def download_model():
    if os.path.exists(MODEL_DIR):
        print("Model already exists. Skipping download.")
        return

    print("Downloading model from Google Drive...")
    gdown.download(URL, ZIP_FILE, quiet=False)

    print("Extracting model...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall()

    os.remove(ZIP_FILE)

    print("Model ready.")


if __name__ == "__main__":
    download_model()