
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ==========================================
# 2. COUNT IMAGES IN EACH CATEGORY
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========\n")

class_counts = {}

for category in CLASSES:
    folder = DATA_DIR / category

    if not folder.exists():
        print(f"Warning: Folder not found: {folder}")
        class_counts[category] = 0
        continue

    images = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower()
        in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    class_counts[category] = len(images)

    print(f"{category}: {len(images)} images")

total_images = sum(class_counts.values())

print(f"\nTotal images: {total_images}")

# Save class distribution report
distribution_df = pd.DataFrame(
    list(class_counts.items()),
    columns=["Category", "Image_Count"]
)

distribution_df.to_csv(
    OUTPUT_DIR / "class_distribution.csv",
    index=False
)

# ==========================================
# 3. CLASS DISTRIBUTION GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("Waste Dataset - Class Distribution")
plt.xlabel("Waste Categories")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png",
    dpi=300
)

plt.close()

# ==========================================
# 4. IMAGE DIMENSION ANALYSIS
# ==========================================

print("\n========== IMAGE DIMENSION ANALYSIS ==========\n")

dimension_records = []

for category in CLASSES:
    folder = DATA_DIR / category

    if not folder.exists():
        continue

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        try:
            with Image.open(file) as img:
                width, height = img.size

                dimension_records.append({
                    "Category": category,
                    "Image_Name": file.name,
                    "Width": width,
                    "Height": height,
                    "Format": img.format,
                    "Aspect_Ratio": round(width / height, 2)
                })

        except Exception as e:
            print(f"Could not read {file}: {e}")

dimension_df = pd.DataFrame(dimension_records)

dimension_df.to_csv(
    OUTPUT_DIR / "image_dimension_analysis.csv",
    index=False
)

if not dimension_df.empty:

    print(dimension_df[["Width", "Height"]].describe())

    print("\nCategory-wise dimension statistics:\n")

    print(
        dimension_df.groupby("Category")[["Width", "Height"]]
        .agg(["min", "max", "mean"])
        .round(2)
    )

else:
    print("No readable images found.")

# ==========================================
# 5. DISPLAY SAMPLE IMAGES
# ==========================================

print("\n========== SAMPLE IMAGE VISUALIZATION ==========\n")

random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for ax, category in zip(axes.flatten(), CLASSES):

    folder = DATA_DIR / category

    if not folder.exists():
        ax.set_title(f"{category} - folder missing")
        ax.axis("off")
        continue

    images = [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower()
        in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    if not images:
        ax.set_title(f"{category} - no images")
        ax.axis("off")
        continue

    sample_image = random.choice(images)

    with Image.open(sample_image) as img:
        ax.imshow(img.convert("RGB"))

    ax.set_title(category.capitalize())
    ax.axis("off")

plt.suptitle("Sample Images from Each Waste Category")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sample_images.png",
    dpi=300
)

plt.close()

# ==========================================
# 6. FINAL EDA SUMMARY
# ==========================================

print("\n========== EDA SUMMARY ==========\n")

print(f"Total images analysed: {total_images}")
print(f"Total categories: {len(CLASSES)}")

if class_counts:
    largest_class = max(class_counts, key=class_counts.get)
    smallest_class = min(class_counts, key=class_counts.get)

    print(
        f"Largest category: {largest_class} "
        f"({class_counts[largest_class]} images)"
    )

    print(
        f"Smallest category: {smallest_class} "
        f"({class_counts[smallest_class]} images)"
    )

print("\nEDA reports saved inside outputs folder.")
print("\nEDA COMPLETED SUCCESSFULLY!")
