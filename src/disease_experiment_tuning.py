from sacred import Experiment
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import numpy as np
from logger import get_logger
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import pickle
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from dotenv import load_dotenv
from datetime import datetime


from disease_data_ingredient import data_ingredient, load_data
from disease_model_ingredient import model_ingredient, get_model, get_param_grid
from disease_preproc_ingredient import (
    preproc_ingredient,
    get_column_transformer,
    get_preproc_param_grid,
)

load_dotenv()

_logs = get_logger(__name__)
ex = Experiment(
    "Disease Experiment",
    ingredients=[data_ingredient, model_ingredient, preproc_ingredient],
)

ex.logger = _logs

timestamp = None

def updatetimestamp():
    global timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
    _logs.info(f"Global timestamp updated to: {timestamp}")

# Example function that uses the global timestamp
def use_timestamp():
    global timestamp
    if timestamp is None:
        _logs.info("Timestamp has not been initialized!")
        return None
    else:
        return timestamp


@ex.config
def cfg():
    """
    Main experiment config.
    """
    models = ["LogisticRegression", "RandomForest", "KNN"]
    # models = ["CustomNeuralNetMDK"]
    folds = 5
    scoring = {"accuracy": "accuracy", "f1": make_scorer(f1_score, average="weighted")}
    refit = "f1"
    # preprocessors = [None, "SelectKBest"]  # removed RFE
    preprocessors = ["RFE"]

@ex.capture
def get_pipe(model, preproc=None, X=None):
    """
    Main pipeline builder: gets a preprocessing name and a classifier name, returns a pipeline.
    """
    _logs.info(f"Getting {model} with {preproc} pre-processing classifier pipeline.")

    clf = get_model(model, X)
    steps = []

    if preproc is not None:
        ct = get_column_transformer(preproc)
        steps.append(("preproc", ct))

    steps.append(("clf", clf))

    pipe = Pipeline(steps=steps)
    return pipe


@ex.capture
def grid_search(pipe, param_grid, X, Y, folds, scoring, refit):
    """Perform grid search on a pipeline given a parameter grid and data."""
    _logs.info(f"Tuning model")

    cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

    print(cv_splitter)

    gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv_splitter, refit=refit)
    gs.fit(X, Y)
    _logs.info(f"Best score: {gs.best_score_}")
    _logs.info(f"Best params: {gs.best_params_}")
    res_dict = gs.cv_results_
    res = pd.DataFrame(res_dict)
    pipe_best = gs.best_estimator_

    # Saving selected features and excluded features for each iteration
    selected_features = []
    excluded_features = []

    # Iterate through the cross-validation folds using the KFold splitter
    for i, (train_idx, val_idx) in enumerate(cv_splitter.split(X, Y)):
        # Get the training and validation data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        # Fit the pipeline for the current fold
        pipe_best.fit(X_train, Y_train)

        # Check if the preprocessor is SelectKBest or RFE and get selected/excluded features
        if "preproc" in pipe_best.named_steps:
            preproc = pipe_best.named_steps["preproc"]

            if isinstance(preproc, (SelectKBest, RFE)):
                selected = X.columns[preproc.get_support()]  # Selected features
                excluded = X.columns[~preproc.get_support()]  # Excluded features

                selected_features.append(selected.tolist())
                excluded_features.append(excluded.tolist())

                # _logs.info(f"Fold {i+1} - Selected features: {selected.tolist()}")
                # _logs.info(f"Fold {i+1} - Excluded features: {excluded.tolist()}")

                # Save feature selection results for all folds
                feature_selection_results = {
                    "fold": np.arange(len(selected_features)) + 1,
                    "selected_features": selected_features,
                    "excluded_features": excluded_features,
                }
                feature_selection_df = pd.DataFrame(feature_selection_results)

                preproc_name = preproc.__class__.__name__
                model = pipe.named_steps["clf"]
                model_name = model.__class__.__name__

                # Save the feature selection results to CSV
                feature_selection_file = f"./reports/feature_selection/{model_name}_{preproc_name}_{use_timestamp()}.csv"
                feature_selection_df.to_csv(feature_selection_file, index=False)

                _logs.info(
                    f"Saved feature selection results to {feature_selection_file}"
                )

            else:
                _logs.warning(
                    f"Fold {i+1} - No feature selection step found in the pipeline"
                )

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Check if the best estimator is a Keras model
    if isinstance(pipe_best, KerasClassifier):
        _logs.info("Detected Keras model, storing training history.")

        # Store the training history for each fold (KerasClassifier history)
        for history in pipe_best.history:
            train_accuracies.append(history["accuracy"])
            val_accuracies.append(history["val_accuracy"])
            train_losses.append(history["loss"])
            val_losses.append(history["val_loss"])

        # Flatten the history and prepare the data to save into a CSV
        history_df = pd.DataFrame(
            {
                "epoch": np.arange(1, len(train_accuracies[0]) + 1),
                "train_accuracy": np.concatenate(train_accuracies),
                "val_accuracy": np.concatenate(val_accuracies),
                "train_loss": np.concatenate(train_losses),
                "val_loss": np.concatenate(val_losses),
            }
        )

        # Create a directory for saving history if it doesn't exist
        history_dir = os.getenv("HISTORY_PATH", "./reports/keras_training")
        os.makedirs(history_dir, exist_ok=True)

        history_file = os.path.join(history_dir, f"history_{use_timestamp()}.csv")

        # Save the history to a CSV file
        history_df.to_csv(history_file, index=False)

        _logs.info(f"Saved training history to {history_file}")

    return res, pipe_best


@ex.capture
def pickle_model_artifact(pipe, model, preprocessor, _run):
    """
    Save model object to disk and add it as an artifact to the experiment run.
    """

    _logs.info(f"Pickling model artifact")

    artifacts_dir = os.getenv("ARTIFACTS_DIR")
    os.makedirs(artifacts_dir, exist_ok=True)

    if isinstance(model, Sequential):  # Keras model
        model_type = "keras"
        # Save the Keras model in SavedModel format (this will save the model architecture, weights, etc.)
        outpath = os.path.join(
            artifacts_dir, f"model_{model}_{preprocessor}_{use_timestamp()}"
        )
        model.save(outpath)  # This will save the Keras model
    else:  # scikit-learn model
        model_type = "sklearn"
        # Save the scikit-learn model using pickle
        outpath = os.path.join(
            artifacts_dir, f"model_{model}_{preprocessor}_{use_timestamp()}.pkl"
        )
        with open(outpath, "wb") as f:
            pickle.dump(pipe, f)

    # Add artifact to experiment run
    _run.add_artifact(outpath)

    _logs.info(f"Pickled model artifact to {outpath}")


@ex.automain
def run(preprocessors, models):
    """Main experiment run."""
    X, Y = load_data()

    for preprocessor in preprocessors:
        for model in models:

            updatetimestamp()

            pipe = get_pipe(model, preprocessor, X)

            model_param_grid = get_param_grid(model)
            preproc_param_grid = get_preproc_param_grid(preprocessor)

            # Merge parameter grids
            param_grid = {}
            if model_param_grid:
                param_grid.update(model_param_grid)
            if preproc_param_grid:
                param_grid.update(preproc_param_grid)

            if param_grid is not None:
                res, pipe_best = grid_search(pipe, param_grid, X, Y)
                _logs.info(f"Optimization results {res.shape}")
                pickle_model_artifact(pipe_best, model, preprocessor)

                res.to_csv(
                    f"./reports/grid_search_results_{model}_{preprocessor}_{use_timestamp()}.csv",
                    index=False,
                )
                _logs.info(
                    f"Grid search results saved to ./reports/grid_search_results_{model}_{preprocessor}_{use_timestamp()}.csv"
                )

            else:
                _logs.warning(f"Parameter grid is None for {model}")


# Commented out cause keeping it runs the experiment twice
# if __name__=="__main__":
#     ex.run_commandline()
