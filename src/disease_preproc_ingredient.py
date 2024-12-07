import pandas as pd
from sacred import Ingredient
from logger import get_logger
from dotenv import load_dotenv
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin

load_dotenv()

_logs = get_logger(__name__)

preproc_ingredient = Ingredient("preproc_ingredient")

preproc_ingredient.logger = _logs


@preproc_ingredient.config
def cfg():
    """
    Config function of the ingredient: all values defined here are shared across captured functions.
    """
    pass


@preproc_ingredient.capture
def get_column_transformer(preproc_pipe):
    """
    Get a column transformer given a name of the preproc_pipe
    """
    _logs.info(f"Getting Column Transformer {preproc_pipe}")

    if preproc_pipe == "SelectKBest":
        return SelectKBest(score_func=chi2)
    elif preproc_pipe == "RFE":
        return RFE(estimator=LogisticRegression())
    else:
        return None


@preproc_ingredient.capture
def get_preproc_param_grid(preproc_pipe):
    """
    Get hyperparameter grid for preprocessors.
    """
    _logs.info(f"Getting parameter grid for preprocessor {preproc_pipe}")
    if preproc_pipe == "SelectKBest":
        return {"preproc__k": [80, 90, 100, 110, 120]}
    elif preproc_pipe == "RFE":
        return {
            "preproc__n_features_to_select": [100, 110, 120],
            "preproc__step": [1, 2],
        }
    else:
        return None
