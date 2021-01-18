import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import pipeline

# Modelling

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
from torch import optim

# model evaluation 
from sklearn import metrics #log_loss, accuracy_score
from sklearn import model_selection # cross_val_score

# hyperparam optimization
import skopt

# custom code
from utils.viz_utils import Diamond
from utils import geometry
from utils.preprocessing import shortstop_global_preprocessing, shortstop_prep_inputs, identity_function
from utils.ml_training import ModelPersistance, ModelExperiment

from __init__ import DATA_DIR

def main():

    # Global Preprocessing to generate global dataset
    df = pd.read_csv(f'{DATA_DIR}/shortstopdefense.csv')
    df = shortstop_global_preprocessing(df)

    feature_columns = [
        'is_runnersgoing',
        'launch_vert_ang',
        'launch_horiz_ang',
        'launch_speed',
        'landing_location_radius',
        'hang_time',
        'landing_location_x',
        'landing_location_y',
        'runner_on_first',
        'runner_on_second',
        'runner_on_third'
    ]

    df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df, feature_columns)

    experiment = ModelExperiment(

        X = X,
        y = y,
        target_name = target_name,
        feature_columns = feature_columns,
        
        model = LogisticRegression(max_iter=10000),
        
        feature_preprocessing = ColumnTransformer([
                ('passthrough', FunctionTransformer(identity_function), ['is_runnersgoing']),
            ], remainder='passthrough'),
        
        space = [
            skopt.space.Real(0.01, 1000, "log-uniform", name='model__C')
            ],
        
        objective = metrics.make_scorer(
            metrics.log_loss,
            needs_proba=True,
            greater_is_better=False
            ),
        
        experiment_description = f"""
            Baseline Logistic Regression model using non-transformed inputs, with no extra features added.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """
    )

    experiment.run_gp_inner_loop(n_calls=20)

if __name__ == '__main__':
    main()