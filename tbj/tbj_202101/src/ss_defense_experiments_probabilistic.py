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
import pygam

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
        'launch_speed',
        'launch_vert_ang',
        'landing_location_radius', # maybe
        'hang_time', # maybe
        'player_time_to_point_of_interest',
        'ball_time_to_point_of_interest', # maybe this should be eliminated b/c is superfluous w/ player_time_to_point_of_interest and player_time_minus_ball_time
        'player_time_minus_ball_time', # always positive, lots of 0s for plays made
        'distance_from_point_of_interest_to_landing_location', # hopefully can distinguish short hops from good hops
        'player_angle_from_interception_point_to_base_of_interest',
        'player_distance_from_interception_point_to_base_of_interest'
    ]

    log_transform_feature_columns = [
        'hang_time',
        'distance_from_point_of_interest_to_landing_location'
    ]
    standard_scalar_feature_columns = [
        'launch_speed',
        'launch_vert_ang',
        'landing_location_radius',
        'hang_time',
        'player_time_to_point_of_interest',
        'ball_time_to_point_of_interest',
        'player_time_minus_ball_time',
        'distance_from_point_of_interest_to_landing_location',
        'player_angle_from_interception_point_to_base_of_interest',
        'player_distance_from_interception_point_to_base_of_interest'
    ]

    df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df, feature_columns)

    # Define params that are consistent between experiments

    param_payload = dict(
        X = X,
        y = y,
        target_name = target_name,
        feature_columns = feature_columns,
        feature_preprocessing = ColumnTransformer([
                ('log_transform', FunctionTransformer(np.log, np.exp), log_transform_feature_columns),
                ('standard_scalar', StandardScaler(), standard_scalar_feature_columns)
            ], remainder='passthrough'),
        objective = metrics.make_scorer(
            metrics.log_loss,
            needs_proba=True,
            greater_is_better=False
            )
    )

    # Logistic Regression

    experiment = ModelExperiment(
        
        model = LogisticRegression(max_iter=10000),
        space = [
            skopt.space.Real(0.01, 1000, "log-uniform", name='model__C')
            ],
        experiment_description = f"""
             Logistic Regression model using fully designed and transformed inputs, all standard scaled, with no extra features added.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_gp_inner_loop(n_calls=25)


    # GAM w/Sigmoid

    # # loop to generate a sum of splines on each feature
    # for i, _ in enumerate(feature_columns):
    #     if i == 0:
    #         gam_features = pygam.s(i)
    #     else:
    #         gam_features += pygam.s(i)

    # experiment = ModelExperiment(
        
    #     model = pygam.LogisticGAM(gam_features, max_iter=1000),
    #     space = [],
    #     experiment_description = f"""
    #          Logistic Generalized Additive Model using fully designed and transformed inputs, all standard scaled. Default for terms 
    #          used is a univariate spline for each feature. No interaction terms.

    #         Input Features: {feature_columns}
    #         Ouptut: {target_name}
    #         """,
    #     **param_payload
    # )
    # experiment.run_single_cross_validation()

if __name__ == '__main__':
    main()