
from pathlib import Path
import json
import gc

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_DIR = Path("data/split/train")
VAL_DIR = Path("data/split/val")

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

# ==========================================
# 2. CHECK DATASET
# ==========================================

if not TRAIN_DIR.exists() or not VAL_DIR.exists():
    raise FileNotFoundError(
        "Training or validation dataset not found. "
        "Run src/split_data.py first."
    )

print("\n========== LOADING DATASET ==========\n")

# ==========================================
# 3. LOAD TRAINING DATA
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nWaste categories:", class_names)
print("Number of classes:", num_classes)

if val_ds.class_names != class_names:
    raise ValueError("Training and validation classes do not match.")

# Save class names
with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# ==========================================
# 5. CLASS WEIGHTS FOR IMBALANCED DATA
# ==========================================

print("\n========== CALCULATING CLASS WEIGHTS ==========\n")

class_counts = {}

for category in class_names:
    folder = TRAIN_DIR / category

    count = sum(
        1 for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]
    )

    class_counts[category] = count

total_train_images = sum(class_counts.values())

class_weights = {}

for index, category in enumerate(class_names):
    count = class_counts[category]

    class_weights[index] = (
        total_train_images / (num_classes * count)
    )

    print(
        f"{category}: {count} images | "
        f"Weight: {class_weights[index]:.3f}"
    )

# ==========================================
# 6. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
], name="data_augmentation")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 7. COMMON CALLBACKS
# ==========================================

def get_callbacks(model_path):

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=0.000001,
            verbose=1
        )
    ]

# ==========================================
# 8. BUILD CNN MODEL
# ==========================================

print("\n========== BUILDING CNN MODEL ==========\n")

cnn_model = models.Sequential([

    layers.Input(shape=(160, 160, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

# ==========================================
# 9. TRAIN CNN MODEL
# ==========================================

print("\n========== TRAINING CNN MODEL ==========\n")

cnn_history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        str(MODEL_DIR / "cnn_best.keras")
    )
)

cnn_model.save(MODEL_DIR / "cnn_final.keras")

print("\nCNN TRAINING COMPLETED!")

# Save CNN training history
cnn_history_df = pd.DataFrame(cnn_history.history)

cnn_history_df.to_csv(
    OUTPUT_DIR / "cnn_training_history.csv",
    index=False
)

del cnn_model
gc.collect()
tf.keras.backend.clear_session()

# ==========================================
# 10. BUILD MOBILENETV2 MODEL
# ==========================================

print("\n========== BUILDING MOBILENETV2 MODEL ==========\n")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

# MobileNetV2 preprocessing: [0,255] to [-1,1]
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

mobilenet_model = models.Model(
    inputs,
    outputs
)

mobilenet_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

mobilenet_model.summary()

# ==========================================
# 11. TRAIN MOBILENETV2 MODEL
# ==========================================

print("\n========== TRAINING MOBILENETV2 MODEL ==========\n")

mobilenet_history = mobilenet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=get_callbacks(
        str(MODEL_DIR / "mobilenet_best.keras")
    )
)

mobilenet_model.save(
    MODEL_DIR / "mobilenet_final.keras"
)

print("\nMOBILENETV2 TRAINING COMPLETED!")

# Save MobileNetV2 training history
mobilenet_history_df = pd.DataFrame(
    mobilenet_history.history
)

mobilenet_history_df.to_csv(
    OUTPUT_DIR / "mobilenet_training_history.csv",
    index=False
)

# ==========================================
# 12. MODEL COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========\n")

cnn_best_accuracy = max(
    cnn_history.history["val_accuracy"]
)

mobilenet_best_accuracy = max(
    mobilenet_history.history["val_accuracy"]
)

print(
    f"CNN Best Validation Accuracy: "
    f"{cnn_best_accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 Best Validation Accuracy: "
    f"{mobilenet_best_accuracy * 100:.2f}%"
)

# Plot validation accuracy
plt.figure(figsize=(10, 6))

plt.plot(
    cnn_history.history["val_accuracy"],
    marker="o",
    label="CNN"
)

plt.plot(
    mobilenet_history.history["val_accuracy"],
    marker="o",
    label="MobileNetV2"
)

plt.title("CNN vs MobileNetV2 - Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=300
)

plt.close()

# Save comparison report
comparison_df = pd.DataFrame({
    "Model": ["CNN", "MobileNetV2"],
    "Best_Validation_Accuracy": [
        cnn_best_accuracy,
        mobilenet_best_accuracy
    ]
})

comparison_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

print("\nComparison graph saved in outputs/")
print("Training history saved in outputs/")
print("Trained models saved in models/")

print("\n========== MODEL TRAINING COMPLETED ==========")
