
import hashlib
import shutil
import csv
from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError

# --------------------------------
# 1. Dataset paths
# --------------------------------

SOURCE_DIR = Path(
    "data/garbage-classification/Garbage classification/Garbage classification"
)

CLEAN_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# --------------------------------
# 2. Calculate image hash
# --------------------------------

def calculate_hash(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------
# 3. Main cleaning function
# --------------------------------

def clean_dataset():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_DIR.resolve()}"
        )

    # Rebuild cleaned dataset on each run
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)

    CLEAN_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    valid_images = []
    corrupted_images = []
    invalid_extensions = []

    # --------------------------------
    # 4. Validate images
    # --------------------------------

    print("\n========== IMAGE VALIDATION ==========\n")

    for category in CLASSES:

        category_dir = SOURCE_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Missing category: {category_dir}"
            )

        files = sorted(category_dir.iterdir())

        for image_path in files:

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                invalid_extensions.append(str(image_path))
                continue

            try:
                with Image.open(image_path) as image:

                    image.verify()

                # Reopen after verify and check dimensions
                with Image.open(image_path) as image:

                    width, height = image.size

                    if width <= 0 or height <= 0:
                        raise ValueError("Invalid dimensions")

                    image_format = image.format

                image_hash = calculate_hash(image_path)

                valid_images.append({
                    "path": image_path,
                    "category": category,
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "hash": image_hash
                })

            except (
                OSError,
                ValueError,
                UnidentifiedImageError
            ) as error:

                corrupted_images.append({
                    "path": str(image_path),
                    "error": str(error)
                })

    # --------------------------------
    # 5. Identify duplicate images
    # --------------------------------

    hash_groups = defaultdict(list)

    for item in valid_images:
        hash_groups[item["hash"]].append(item)

    unique_images = []
    duplicate_images = []

    for image_hash, group in hash_groups.items():

        categories = {
            item["category"] for item in group
        }

        # If the same image has different labels,
        # exclude all copies for manual review.
        if len(categories) > 1:

            for item in group:
                duplicate_images.append({
                    "path": str(item["path"]),
                    "category": item["category"],
                    "reason": "Same image found in different categories"
                })

            continue

        # Keep only one copy within the same category
        unique_images.append(group[0])

        for duplicate in group[1:]:
            duplicate_images.append({
                "path": str(duplicate["path"]),
                "category": duplicate["category"],
                "reason": "Exact duplicate within category"
            })

    # --------------------------------
    # 6. Create cleaned dataset
    # --------------------------------

    print("\n========== COPYING CLEAN IMAGES ==========\n")

    class_counts = defaultdict(int)
    dimensions = []

    for item in unique_images:

        category = item["category"]

        destination_dir = CLEAN_DIR / category
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / item["path"].name

        shutil.copy2(item["path"], destination)

        class_counts[category] += 1

        dimensions.append({
            "filename": item["path"].name,
            "category": category,
            "width": item["width"],
            "height": item["height"],
            "format": item["format"]
        })

    # --------------------------------
    # 7. Save cleaning reports
    # --------------------------------

    with open(
        OUTPUT_DIR / "cleaning_report.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Category", "Clean Images"])

        for category in CLASSES:
            writer.writerow([
                category,
                class_counts[category]
            ])

    with open(
        OUTPUT_DIR / "image_dimensions.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "category",
                "width",
                "height",
                "format"
            ]
        )

        writer.writeheader()
        writer.writerows(dimensions)

    with open(
        OUTPUT_DIR / "cleaning_issues.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Image Path", "Category", "Issue"])

        for item in corrupted_images:
            writer.writerow([
                item["path"],
                "",
                "Corrupted: " + item["error"]
            ])

        for item in duplicate_images:
            writer.writerow([
                item["path"],
                item["category"],
                item["reason"]
            ])

        for item in invalid_extensions:
            writer.writerow([
                item,
                "",
                "Unsupported image extension"
            ])

    # --------------------------------
    # 8. Print results
    # --------------------------------

    print("\n========== CLEANING SUMMARY ==========\n")

    print("Original valid images:", len(valid_images))
    print("Clean images:", len(unique_images))
    print("Corrupted images:", len(corrupted_images))
    print("Duplicate images:", len(duplicate_images))
    print("Unsupported files:", len(invalid_extensions))

    print("\nCategory-wise clean image counts:\n")

    for category in CLASSES:
        print(f"{category}: {class_counts[category]}")

    print("\nCleaned dataset saved at:", CLEAN_DIR)
    print("Cleaning reports saved in:", OUTPUT_DIR)

    print("\nDATA CLEANING COMPLETED!")


if __name__ == "__main__":
    clean_dataset()

