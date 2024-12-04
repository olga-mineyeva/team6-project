from sacred import Experiment
import os
import pickle
import matplotlib.pyplot as plt
from logger import get_logger
from dotenv import load_dotenv

from disease_data_ingredient import data_ingredient, load_data, load_validation_data

load_dotenv()

ex  = Experiment("Disease Experiment Evaluation", ingredients=[data_ingredient])

@ex.config
def cfg():
    pass

@ex.automain
def run():
# Load the model
    with open('./models/model_RandomForest_None.pkl', 'rb') as file:
        model = pickle.load(file)

    X_train, Y_train = load_data()
    X_val, Y_val = load_validation_data()

    # Evaluate on training and validation sets
    train_accuracy = model.score(X_train, Y_train)
    val_accuracy = model.score(X_val, Y_val)

    # Print the results
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Validation Accuracy: {val_accuracy}")

    # Visualize

    # Data for visualization
    data = {'Training': train_accuracy, 'Validation': val_accuracy}
    names = list(data.keys())
    values = list(data.values())

    plt.bar(names, values, color=['blue', 'orange'])
    plt.ylim(0, 1)  # Ensure the y-axis represents accuracy
    plt.title("Training vs Validation Accuracy")
    plt.ylabel("Accuracy")
    plt.show()
