from sacred import Experiment
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from disease_data_ingredient import data_ingredient, load_data, load_validation_data

load_dotenv()
_logs = get_logger(__name__)

ex = Experiment("Disease Experiment Evaluation", ingredients=[data_ingredient])

ex.logger = _logs

@ex.config
def cfg():
    pass


@ex.automain
def run():
    # Load the model
    models_dir = "./models"

    # Get a list of all .pkl files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    X_train, Y_train = load_data()
    X_val, Y_val = load_validation_data()

    # Ensure binary labels Y/N (convert if needed)
    unique_labels = np.unique(Y_val)
    assert len(unique_labels) == 2, "This script assumes binary classification (Y/N)."

    # Map labels to 0 and 1 for consistency if not already done
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    Y_val_mapped = np.array([label_map[label] for label in Y_val])

    # Initialize overall confusion matrix
    overall_cm = np.zeros((2, 2), dtype=int)

    evaluation_results = {}

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Evaluate on training and validation sets
        train_accuracy = model.score(X_train, Y_train)
        val_accuracy = model.score(X_val, Y_val)

        # Get the model parameters
        model_params = model.get_params() if hasattr(model, "get_params") else {}

        # Get preprocessing steps if using a pipeline
        preprocessing_steps = ""
        if hasattr(model, "named_steps"):
            preprocessing_steps = [name for name, step in model.named_steps.items() if name != "clf"]

        # Store the results
        evaluation_results[model_file] = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "model_params": model_params,
            "preprocessing_steps": preprocessing_steps,
        }

        # Predict and calculate confusion matrix for binary Y/N
        Y_pred = model.predict(X_val)
        Y_pred_mapped = np.array([label_map[label] for label in Y_pred])

        cm = confusion_matrix(Y_val_mapped, Y_pred_mapped, labels=[0, 1])  # Binary format
        overall_cm += cm  # Aggregate to overall confusion matrix

        # Log results and plot confusion matrix for this model
        _logs.info(f"Evaluating model: {model_file}")
        _logs.info(f"Training Accuracy: {train_accuracy}")
        _logs.info(f"Validation Accuracy: {val_accuracy}")
        _logs.info(f"Confusion Matrix:\n{cm}")
        _logs.info(f"Model Parameters: {model_params}")
        _logs.info(f"Preprocessing Steps: {preprocessing_steps}\n")

        # Plot confusion matrix for this model
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["N", "Y"], yticklabels=["N", "Y"])
        plt.title(f"Confusion Matrix for {model_file}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save the confusion matrix plot
        plot_dir = "./reports/confusion_matrices"
        os.makedirs(plot_dir, exist_ok=True)
        plot_file = os.path.join(plot_dir, f"confusion_matrix_{model_file}.png")
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

    # Plot overall confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["N", "Y"], yticklabels=["N", "Y"])
    plt.title("Overall Confusion Matrix Across All Models")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the overall confusion matrix plot
    overall_plot_file = os.path.join(plot_dir, "overall_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(overall_plot_file)
    plt.close()

    _logs.info(f"Overall confusion matrix saved as {overall_plot_file}")
