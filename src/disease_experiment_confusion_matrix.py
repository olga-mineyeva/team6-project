from sacred import Experiment
import os
import pickle
import matplotlib.pyplot as plt
from logger import get_logger
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from disease_data_ingredient import data_ingredient, load_data, load_validation_data

load_dotenv()

ex = Experiment("Disease Experiment Evaluation", ingredients=[data_ingredient])

@ex.config
def cfg():
    pass

@ex.automain
def run():
    models_dir = "./models"

    # Get a list of all .pkl files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

    evaluation_results = {}
    
    # Inside the loop where models are evaluated:
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        X_train, Y_train = load_data()
        X_val, Y_val = load_validation_data()

        # Evaluate on training and validation sets
        train_accuracy = model.score(X_train, Y_train)
        val_accuracy = model.score(X_val, Y_val)

        # Generate predictions for validation set
        Y_val_pred = model.predict(X_val)

        # Compute the confusion matrix
        cm = confusion_matrix(Y_val, Y_val_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size as needed for clarity
        disp.plot(ax=ax, cmap="viridis", values_format="d")

        plt.xticks(rotation=90)

        plot_dir = "./reports/confusion_matrices"
        os.makedirs(plot_dir, exist_ok=True)

        # Save confusion matrix plot to file
        cm_plot_file = os.path.join(plot_dir, f"confusion_matrix_{model_file}.png")
        plt.title(f"Confusion Matrix for {model_file}")
        plt.savefig(cm_plot_file)
        plt.close()

        # Store the results
        evaluation_results[model_file] = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "confusion_matrix": cm,
        }

        # Print the results for each model
        print(f"Evaluating model: {model_file}")
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}\n")
        print(f"Confusion matrix saved to: {cm_plot_file}\n")
