from sacred import Experiment
import pandas as pd
from logger import get_logger

from disease_data_ingredient import data_ingredient, load_data

_logs = get_logger(__name__)
ex  = Experiment("Disease Experiment",
                 ingredients=[data_ingredient])

ex.logger = _logs

@ex.automain
def run():
    '''Main experiment run.'''
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    # print(X.columns.tolist())
    # print(Y.columns.tolist())

# Commented out cause keeping it runs the experiment twice
# if __name__=="__main__":
#     ex.run_commandline()