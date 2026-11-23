
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

TEST_DIR = Path("data/split/test")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

CNN_PATH = MODEL_DIR / "cnn_best.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenet_best.keras"

# ==========================================
# 2. CHECK FILES
# ==========================================

for path in [TEST_DIR, CNN_PATH, MOBILENET_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file or folder not found: {path}"
        )

# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

print("\n========== LOADING TEST DATASET ==========\n")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("Categories:", class_names)

# Extract actual labels
y_true = np.concatenate([
    labels.numpy() for images, labels in test_ds
])

print("Total test images:", len(y_true))

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================

def evaluate_model(model_path, model_name):

    print(f"\n========== {model_name} EVALUATION ==========\n")

    # Load best saved model
    model = tf.keras.models.load_model(model_path)

    # Evaluate test loss and accuracy
    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    # Predict test images
    predictions = model.predict(test_ds, verbose=1)

    y_pred = np.argmax(predictions, axis=1)

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )

    print("\nClassification Report:\n")
    print(report)

    # Save classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    report_df.to_csv(
        OUTPUT_DIR / f"{model_name.lower()}_classification_report.csv"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes))
    )

    print("\nConfusion Matrix:\n")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{model_name.lower()}_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    del model

    return {
        "Model": model_name,
        "Test_Accuracy": accuracy,
        "Test_Loss": test_loss
    }


# ==========================================
# 5. EVALUATE CNN
# ==========================================

cnn_result = evaluate_model(
    CNN_PATH,
    "CNN"
)

tf.keras.backend.clear_session()

# ==========================================
# 6. EVALUATE MOBILENETV2
# ==========================================

mobilenet_result = evaluate_model(
    MOBILENET_PATH,
    "MobileNetV2"
)

tf.keras.backend.clear_session()

# ==========================================
# 7. FINAL MODEL COMPARISON
# ==========================================

print("\n========== FINAL MODEL COMPARISON ==========\n")

comparison = pd.DataFrame([
    cnn_result,
    mobilenet_result
])

print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "test_model_comparison.csv",
    index=False
)

best_model = comparison.loc[
    comparison["Test_Accuracy"].idxmax()
]

print("\nBest model based on test accuracy:")
print(best_model["Model"])

print(
    f"Test Accuracy: "
    f"{best_model['Test_Accuracy'] * 100:.2f}%"
)

print("\nEvaluation reports saved in outputs/")
print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY!")

