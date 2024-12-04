from sacred import Experiment
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
from logger import get_logger
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import pickle
import os
from dotenv import load_dotenv

from disease_data_ingredient import data_ingredient, load_data
from disease_model_ingredient import model_ingredient, get_model, get_param_grid

load_dotenv()

_logs = get_logger(__name__)
ex  = Experiment("Disease Experiment",
                 ingredients=[data_ingredient, model_ingredient])

ex.logger = _logs

@ex.config
def cfg():
    '''
    Main experiment config.
    '''
    models = ["LogisticRegression", "RandomForest", "KNN", "NeuralNet"]
    folds = 5
    scoring = {'accuracy': 'accuracy', 'f1': make_scorer(f1_score, average='weighted')}
    refit = 'f1' 

@ex.capture
def get_pipe(model):
    '''
    Main pipeline builder: gets a preprocessing name and a classifier name, returns a pipeline.
    '''
    _logs.info(f'Getting {model} classifier pipeline.')
    clf = get_model(model)
    pipe = Pipeline(
        steps  = [
            ('clf', clf)
        ]
    )
    return pipe

@ex.capture
def grid_search(pipe, param_grid, X, Y, folds, scoring, refit):
    '''Perform grid search on a pipeline given a parameter grid and data.'''
    _logs.info(f'Tuning model')
    gs = GridSearchCV(pipe, param_grid, scoring=scoring, cv = folds, refit=refit)
    gs.fit(X, Y)
    _logs.info(f'Best score: {gs.best_score_}')
    _logs.info(f'Best params: {gs.best_params_}')
    res_dict = gs.cv_results_
    res = pd.DataFrame(res_dict)
    pipe_best = gs.best_estimator_
    return res, pipe_best

@ex.capture
def pickle_model_artifact(pipe, model, _run):
    '''
    Save model object to disk and add it as an artifact to the experiment run.
    '''

    _logs.info(f'Pickling model artifact')
    
    artifacts_dir = os.getenv('ARTIFACTS_DIR')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    outpath = os.path.join(
        artifacts_dir, 
        f"model_{model}_{_run._id}.pkl")
    
    with open(outpath, 'wb') as f:
        pickle.dump(pipe, f)
    # Add artifact to experiment run
    _run.add_artifact(outpath)
    
    _logs.info(f'Pickled model artifact to {outpath}')

@ex.automain
def run(models):
    '''Main experiment run.'''
    X, Y  = load_data()

    for model in models:

        pipe = get_pipe(model)
        param_grid = get_param_grid(model)
        if param_grid is not None:
            res, pipe_best = grid_search(pipe, param_grid, X, Y)   
            _logs.info(f'Optimization results {res.shape}')
            pickle_model_artifact(pipe_best, model)

            res.to_csv(f"./reports/grid_search_results_{model}.csv", index=False)
            _logs.info("Grid search results saved to ./reports/grid_search_results.csv")

        else:
            _logs.warning(f'Parameter grid is None for {model}')

# Commented out cause keeping it runs the experiment twice
# if __name__=="__main__":
#     ex.run_commandline()