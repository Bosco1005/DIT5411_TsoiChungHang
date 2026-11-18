# prepare_data.py
import os
import shutil
from pathlib import Path
import random
from config import MAX_CHARS, TRAIN_SAMPLES_PER_CLASS

RAW_DIR = Path("data_raw/cleaned_data")  # adjust if needed
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")

def prepare():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    char_folders = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if MAX_CHARS:
        char_folders = char_folders[:MAX_CHARS]

    print(f"Using {len(char_folders)} characters")

    for char_folder in char_folders:
        images = sorted(list(char_folder.glob("*.*")))
        if len(images) < 45:
            continue

        train_imgs = images[:TRAIN_SAMPLES_PER_CLASS]
        test_imgs = images[TRAIN_SAMPLES_PER_CLASS:]

        (TRAIN_DIR / char_folder.name).mkdir(exist_ok=True)
        (TEST_DIR / char_folder.name).mkdir(exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, TRAIN_DIR / char_folder.name / img.name)
        for img in test_imgs:
            shutil.copy(img, TEST_DIR / char_folder.name / img.name)

    print("Data split done.")

if __name__ == "__main__":
    prepare()