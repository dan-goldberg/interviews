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

    # # Support Vector Classifier w/ RBF Kernel

    # experiment = ModelExperiment(
        
    #     model = SVC(
    #         probability=True,
    #         kernel='rbf'
    #         ),
    #     space = [
    #         skopt.space.Real(0.01, 1000, "log-uniform", name='model__C'),
    #         skopt.space.Real(1e-5, 1e-2, "log-uniform", name='model__tol')
    #         ],
    #     experiment_description = f"""
    #          Support Vector Machine Classifier w/RBF kernel, using fully designed and transformed inputs, all standard scaled.

    #         Input Features: {feature_columns}
    #         Ouptut: {target_name}
    #         """,
    #     **param_payload
    # )
    # experiment.run_gp_inner_loop(n_calls=30)

    # # Support Vector Classifier w/ Polynomial Kernel

    # experiment = ModelExperiment(
        
    #     model = SVC(
    #         probability=True,
    #         kernel='poly'
    #         ),
    #     space = [
    #         skopt.space.Real(0.01, 1000, "log-uniform", name='model__C'),
    #         skopt.space.Real(1e-5, 1e-2, "log-uniform", name='model__tol'),
    #         skopt.space.Integer(2, 4, "uniform", name='model__degree')
    #         ],
    #     experiment_description = f"""
    #          Support Vector Machine Classifier w/Polynomial kernel, using fully designed and transformed inputs, all standard scaled.

    #         Input Features: {feature_columns}
    #         Ouptut: {target_name}
    #         """,
    #     **param_payload
    # )
    # experiment.run_gp_inner_loop(n_calls=30)

    # ### Random Forest

    # experiment = ModelExperiment(
        
    #     model = RandomForestClassifier(),
    #     space = [
    #         skopt.space.Integer(25, 2500, "log-uniform", name='model__n_estimators'),
    #         skopt.space.Integer(2, 20, "uniform", name='model__min_samples_split')
    #         ],
    #     experiment_description = f"""
    #          Random Forest Classifier model using fully designed and transformed inputs, all standard scaled.

    #         Input Features: {feature_columns}
    #         Ouptut: {target_name}
    #         """,
    #     **param_payload
    # )
    # experiment.run_gp_inner_loop(n_calls=30)

    ## Gradient Boosted Trees Classifier

    experiment = ModelExperiment(
        model = xgb.XGBClassifier(),
        space = [
            skopt.space.Integer(25, 2500, "log-uniform", name='model__n_estimators'),
            skopt.space.Integer(3, 15, "uniform", name='model__max_depth'),
            skopt.space.Real(0.0, 1.0, "uniform", name='model__eta')
            ],
        experiment_description = f"""
             Gradient Boosted Trees Classifier model using fully designed and transformed inputs, all standard scaled.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_gp_inner_loop(n_calls=25)

if __name__ == '__main__':
    main()