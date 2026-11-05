
import os
import streamlit as st
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import CropDiseaseCNN


# ==========================================
# 1. STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Health Model Evaluation",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Crop Health Analysis")
st.subheader("CNN Model Performance Evaluation")


# ==========================================
# 2. DEVICE AND PATH CONFIGURATION
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = r"C:\Users\simra\models\best_crop_model.pth"

# Test folder inside the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAL_PATH = r"C:\Users\simra\Downloads\archive (19)\PlantVillage\val"


st.write("Running device:", DEVICE)
st.write("Model path:", MODEL_PATH)


# ==========================================
# 3. CHECK MODEL AVAILABILITY
# ==========================================

if not os.path.exists(MODEL_PATH):

    st.error("Trained model checkpoint not found!")

    st.write("Expected location:", MODEL_PATH)

    st.stop()


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        # Load checkpoint information

        num_classes = checkpoint["num_classes"]

        class_names = checkpoint["class_names"]

        image_size = checkpoint["image_size"]

        best_accuracy = checkpoint["best_val_accuracy"]

        # Initialize original CNN architecture

        model = CropDiseaseCNN(
            num_classes=num_classes
        )

        # Load trained weights

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(DEVICE)

        model.eval()

        return model, class_names, image_size, best_accuracy

    except Exception as e:

        st.error(f"Model loading failed: {e}")

        st.stop()


# Load model only once

model, class_names, IMAGE_SIZE, best_accuracy = load_model()


st.success("Trained CNN model loaded successfully!")


# ==========================================
# 5. DISPLAY MODEL INFORMATION
# ==========================================

st.header("Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Number of Classes",
    len(class_names)
)

col2.metric(
    "Best Validation Accuracy",
    f"{best_accuracy:.2f}%"
)

col3.metric(
    "Execution Device",
    str(DEVICE)
)


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

# IMPORTANT:
# Use exactly the same preprocessing and
# normalization used during model training.

if isinstance(IMAGE_SIZE, int):

    resize_size = (IMAGE_SIZE, IMAGE_SIZE)

else:

    resize_size = tuple(IMAGE_SIZE)


test_transform = transforms.Compose([

    transforms.Resize(resize_size),

    transforms.ToTensor()

])


# ==========================================
# 7. TEST DATASET EVALUATION
# ==========================================

st.header("Test Dataset Evaluation")

st.write(
    "Evaluate your trained CNN model "
    "using unseen crop disease images."
)


if st.button("Evaluate Model Accuracy"):

    if not os.path.isdir(VAL_PATH):

        st.error(
            f"Test dataset folder not found: {VAL_PATH}"
        )

        st.stop()

    # Load test dataset

    test_dataset = datasets.ImageFolder(
        root=VAL_PATH,
        transform=test_transform
    )

    if len(test_dataset) == 0:

        st.error("Test dataset is empty!")

        st.stop()

    # Check classes

    if set(test_dataset.classes) != set(class_names):

        st.error(
            "Test dataset classes do not match "
            "the saved model classes."
        )

        st.write("Model classes:", class_names)

        st.write("Test classes:", test_dataset.classes)

        st.stop()

    # Convert test dataset labels into
    # the model's original class indices.

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    # Create DataLoader

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    actual_labels = []

    predicted_labels = []

    # ======================================
    # 8. MODEL PREDICTION
    # ======================================

    st.subheader("Evaluating Test Images")

    progress = st.progress(0)

    status = st.empty()

    model.eval()

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(test_loader):

            images = images.to(DEVICE)

            # Forward propagation

            outputs = model(images)

            # Get predicted classes

            predictions = torch.argmax(
                outputs,
                dim=1
            ).cpu()

            # Map labels to model class indices

            mapped_labels = [

                class_to_index[
                    test_dataset.classes[label]
                ]

                for label in labels.tolist()

            ]

            actual_labels.extend(mapped_labels)

            predicted_labels.extend(
                predictions.tolist()
            )

            # Update progress bar

            progress.progress(
                (batch_index + 1) / len(test_loader)
            )

            status.text(
                f"Processed {min((batch_index + 1) * 32, len(test_dataset))}"
                f" / {len(test_dataset)} images"
            )

    # ======================================
    # 9. CALCULATE TEST ACCURACY
    # ======================================

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    correct = sum(

        actual == predicted

        for actual, predicted in zip(
            actual_labels,
            predicted_labels
        )

    )

    total = len(actual_labels)

    incorrect = total - correct

    st.success("Model evaluation completed successfully!")

    st.header("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Correct Predictions",
        correct
    )

    col3.metric(
        "Incorrect Predictions",
        incorrect
    )

    col4.metric(
        "Total Test Images",
        total
    )

    # ======================================
    # 10. CLASSIFICATION REPORT
    # ======================================

    st.header("Classification Report")

    report = classification_report(

        actual_labels,

        predicted_labels,

        labels=list(range(len(class_names))),

        target_names=class_names,

        output_dict=True,

        zero_division=0

    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # ======================================
    # 11. CONFUSION MATRIX
    # ======================================

    st.header("Confusion Matrix")

    cm = confusion_matrix(

        actual_labels,

        predicted_labels,

        labels=list(range(len(class_names)))

    )

    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Greens",

        xticklabels=class_names,

        yticklabels=class_names,

        ax=ax

    )

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_title("Crop Disease Classification Confusion Matrix")

    plt.xticks(rotation=90)

    plt.yticks(rotation=0)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ======================================
    # 12. PREDICTION RESULTS
    # ======================================

    st.header("Individual Prediction Results")

    results_df = pd.DataFrame({

        "Actual Class": [

            class_names[i]

            for i in actual_labels

        ],

        "Predicted Class": [

            class_names[i]

            for i in predicted_labels

        ],

        "Result": [

            "Correct" if a == p else "Incorrect"

            for a, p in zip(
                actual_labels,
                predicted_labels
            )

        ]

    })

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # Download results

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="Download Prediction Results",

        data=csv,

        file_name="crop_prediction_results.csv",

        mime="text/csv"

    )

    st.balloons()
