from sacred import Experiment
import os
import pickle
import matplotlib.pyplot as plt
from logger import get_logger
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

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

        Y_pred = model.predict(X_val)

        val_report = classification_report(Y_val, Y_pred, output_dict=True)

        # Store the results
        evaluation_results[model_file] = {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_classification_report": val_report,
        }

        # Print the results for each model
        print(f"Evaluating model: {model_file}")
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Validation Accuracy: {val_accuracy}")
        print(f"Validation Classification Report:\n{val_report}\n")
