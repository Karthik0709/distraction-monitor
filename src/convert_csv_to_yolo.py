import csv
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

# This file lives at <repo_root>/src/convert_csv_to_yolo.py, so the repo root is one level up.
SRC_DIR = Path(__file__).resolve().parent

# --- Fill these in before running ---
# These two point OUTSIDE this repo (your own local dataset download) - not
# something a relative path can express. Edit them for your machine.
IMAGE_DIR = "C:\\Users\\Jayakumar\\Downloads\\archive\\positive"          # uncropped source images
CSV_PATH = "C:\\Users\\Jayakumar\\Downloads\\archive\\labels.csv"         # filename,width,height,xmin,ymin,xmax,ymax,class
ACCEPTED_CLASS = "Comp"                                                    # which CSV 'class' value to keep
OUTPUT_LABEL = "phone"                                                     # class name written into data.yaml
OUTPUT_DIR = str(SRC_DIR / "phone_yolo_dataset")
VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42


def load_annotations(csv_path, accepted_class):
    """Returns {filename: [(xmin, ymin, xmax, ymax), ...]} - one entry per image,
    every box on that image, filtered to accepted_class."""
    annotations = defaultdict(list)
    skipped_other_class = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class"] != accepted_class:
                skipped_other_class += 1
                continue
            annotations[row["filename"]].append((
                int(row["xmin"]), int(row["ymin"]),
                int(row["xmax"]), int(row["ymax"]),
            ))

    if skipped_other_class:
        logger.info("Skipped %d CSV row(s) not matching class '%s'.", skipped_other_class, accepted_class)

    return annotations


def to_yolo_line(class_idx, xmin, ymin, xmax, ymax, img_width, img_height):
    """YOLO format: class_idx cx cy w h, all normalized to [0, 1]."""
    box_w = xmax - xmin
    box_h = ymax - ymin
    cx = xmin + box_w / 2
    cy = ymin + box_h / 2
    return f"{class_idx} {cx / img_width:.6f} {cy / img_height:.6f} {box_w / img_width:.6f} {box_h / img_height:.6f}"


def build_split(filenames, annotations, image_dims, image_dir, images_out_dir, labels_out_dir, class_idx):
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    written = 0
    for filename in filenames:
        src_image_path = os.path.join(image_dir, filename)
        if not os.path.exists(src_image_path):
            logger.warning("Image referenced in CSV not found on disk, skipping: %s", src_image_path)
            continue

        shutil.copy2(src_image_path, os.path.join(images_out_dir, filename))

        img_width, img_height = image_dims[filename]
        lines = [
            to_yolo_line(class_idx, xmin, ymin, xmax, ymax, img_width, img_height)
            for xmin, ymin, xmax, ymax in annotations[filename]
        ]

        base_name = os.path.splitext(filename)[0]
        with open(os.path.join(labels_out_dir, base_name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        written += 1

    return written


def write_data_yaml(output_dir, class_name):
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  0: {class_name}\n")
    return yaml_path


def convert(image_dir, csv_path, accepted_class, output_label, output_dir,
            validation_fraction=VALIDATION_FRACTION, seed=RANDOM_SEED):
    annotations = load_annotations(csv_path, accepted_class)

    image_dims = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["filename"] in annotations:
                image_dims[row["filename"]] = (int(row["width"]), int(row["height"]))

    filenames = list(annotations.keys())
    logger.info("Found %d images with at least one '%s' box.", len(filenames), accepted_class)

    random.Random(seed).shuffle(filenames)
    split_idx = max(1, int(len(filenames) * (1 - validation_fraction)))
    train_filenames = filenames[:split_idx]
    val_filenames = filenames[split_idx:]

    class_idx = 0  # single class - phone is the only thing this model needs to know about

    train_written = build_split(
        train_filenames, annotations, image_dims, image_dir,
        os.path.join(output_dir, "images", "train"),
        os.path.join(output_dir, "labels", "train"),
        class_idx,
    )
    val_written = build_split(
        val_filenames, annotations, image_dims, image_dir,
        os.path.join(output_dir, "images", "val"),
        os.path.join(output_dir, "labels", "val"),
        class_idx,
    )

    yaml_path = write_data_yaml(output_dir, output_label)

    logger.info(
        "Done. train=%d images, val=%d images, written to %s. data.yaml at %s",
        train_written, val_written, output_dir, yaml_path,
    )


def main():
    convert(IMAGE_DIR, CSV_PATH, ACCEPTED_CLASS, OUTPUT_LABEL, OUTPUT_DIR)


if __name__ == "__main__":
    main()
