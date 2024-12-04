import pandas as pd
from sacred import Ingredient
from logger import get_logger
from sklearn.pipeline import Pipeline

_logs = get_logger(__name__)

preproc_ingredient = Ingredient('preproc_ingredient')

preproc_ingredient.logger = _logs
