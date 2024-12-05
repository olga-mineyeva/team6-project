from sacred import Experiment
import os
import pickle
import matplotlib.pyplot as plt
from logger import get_logger
import numpy as np
from dotenv import load_dotenv

from disease_data_ingredient import data_ingredient, load_data, load_validation_data

load_dotenv()

ex = Experiment("Disease Experiment Evaluation", ingredients=[data_ingredient])


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

    evaluation_results = {}

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        # Evaluate on training and validation sets
        train_accuracy = model.score(X_train, Y_train)
        val_accuracy = model.score(X_val, Y_val)

        # Store the results
        evaluation_results[model_file] = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
        }

        # Print the results for each model
        print(f"Evaluating model: {model_file}")
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}\n")

    # Visualization of training vs validation accuracy for all models
    model_names = list(evaluation_results.keys())
    train_accuracies = [
        evaluation_results[model]["train_accuracy"] for model in model_names
    ]
    val_accuracies = [
        evaluation_results[model]["val_accuracy"] for model in model_names
    ]

    # Create bar plots for each model
    x = np.arange(len(model_names))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, train_accuracies, width, label="Training", color="blue")
    ax.bar(x + width / 2, val_accuracies, width, label="Validation", color="orange")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy for Each Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
    ax.legend()

    plt.tight_layout()
    plt.show()
