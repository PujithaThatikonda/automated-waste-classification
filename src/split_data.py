
from pathlib import Path
import shutil
import random

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

SOURCE_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("data/split")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ==========================================
# 2. CHECK SOURCE DATASET
# ==========================================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {SOURCE_DIR}"
    )

for category in CLASSES:
    if not (SOURCE_DIR / category).exists():
        raise FileNotFoundError(
            f"Missing category folder: {category}"
        )

# ==========================================
# 3. CREATE OUTPUT FOLDERS
# ==========================================

# Remove only the previous split dataset.
# The original and cleaned datasets remain safe.
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:
    for category in CLASSES:
        (OUTPUT_DIR / split / category).mkdir(
            parents=True,
            exist_ok=True
        )

# ==========================================
# 4. SPLIT IMAGES CATEGORY-WISE
# ==========================================

print("\n========== DATASET SPLITTING ==========\n")

total_train = 0
total_val = 0
total_test = 0

for category in CLASSES:

    category_path = SOURCE_DIR / category

    images = [
        file for file in category_path.iterdir()
        if file.is_file()
        and file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    test_count = total - train_count - val_count

    train_images = images[:train_count]

    val_images = images[
        train_count:train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    # Copy training images
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / category / image.name
        )

    # Copy validation images
    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / category / image.name
        )

    # Copy testing images
    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / category / image.name
        )

    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)

    print(f"{category.capitalize()}:")
    print(f"  Total: {total}")
    print(f"  Train: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Test: {len(test_images)}")
    print()

# ==========================================
# 5. FINAL SUMMARY
# ==========================================

print("========== FINAL SPLIT SUMMARY ==========\n")

print(f"Training images:   {total_train}")
print(f"Validation images: {total_val}")
print(f"Testing images:    {total_test}")

print(
    f"\nTotal images: {total_train + total_val + total_test}"
)

print("\nDataset split saved inside data/split/")
print("\nDATASET SPLITTING COMPLETED SUCCESSFULLY!")

