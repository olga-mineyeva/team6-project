from sacred import Ingredient
from dotenv import load_dotenv
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from logger import get_logger
import json
import os

load_dotenv()
_logs = get_logger(__name__)

model_ingredient = Ingredient("model_ingredient")
model_ingredient.logger = _logs


@model_ingredient.config
def cfg():
    pass


@model_ingredient.capture
def get_model(model, X=None):
    """
    Given a name, return a model.
    """
    _logs.info(f"Getting model {model}")
    if model == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression()
    elif model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier()
    elif model == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier()
    elif model == "NeuralNet":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier()
    elif model == "CustomNeuralNetMDK":
        from CustomNeuralNetMDKClassifier import CustomNeuralNetMDKClassifier
        return CustomNeuralNetMDKClassifier()
    else:
        return None


def get_param_grid(model):
    """
    Given a name, return a parameter grid. Param grids are stored in json files.
    """
    _logs.info(f"Getting parameter grid for {model}")
    file = None
    if model == "LogisticRegression":
        file = os.getenv("LOGISTIC_REGRESSION_PG")
    elif model == "RandomForest":
        file = os.getenv("RANDOM_FOREST_PG")
    elif model == "KNN":
        file = os.getenv("KNN_PG")
    elif model == "NeuralNet":
        file = os.getenv("NEURAL_NET_PG")
    elif model == "CustomNeuralNetMDK":
        file = os.getenv("CUSTOM_NEURAL_NET_MDK3_PG")

    if file is not None:
        with open(file) as f:
            return json.load(f)
    else:
        return None


def MDKMLClassifierv3(input_dim, num_classes=41, layers=[128, 64], optimizer="adam"):
    model = Sequential()
    # Input layer + first hidden layer
    model.add(Dense(layers[0], input_dim=input_dim, activation="relu"))
    # Second hidden layer
    model.add(Dense(layers[1], activation="relu"))
    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def wrapped_MDKMLClassifierv3(input_dim):
    return KerasClassifier(
        build_fn=MDKMLClassifierv3,
        input_dim=input_dim,
        num_classes=41,
        epochs=20,
        batch_size=32,
        verbose=0,
    )
