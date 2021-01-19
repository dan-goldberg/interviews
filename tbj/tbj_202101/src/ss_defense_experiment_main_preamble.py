import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import pipeline

# model evaluation 
from sklearn import metrics #log_loss, accuracy_score
from sklearn import model_selection # cross_val_score

# custom code
from utils.preprocessing import shortstop_global_preprocessing, shortstop_prep_inputs
from utils.preprocessing import LogScaleTransformer

from __init__ import DATA_DIR

def experiment_prep():

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

    log_scale_transform_feature_columns = [
        'hang_time',
        'distance_from_point_of_interest_to_landing_location'
    ]
    standard_scalar_feature_columns = [
        'launch_speed',
        'launch_vert_ang',
        'landing_location_radius',
        'player_time_to_point_of_interest',
        'ball_time_to_point_of_interest',
        'player_time_minus_ball_time',
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
                ('log_scale_transform', LogScaleTransformer(), log_scale_transform_feature_columns),
                ('standard_scalar', StandardScaler(), standard_scalar_feature_columns)
            ], remainder='passthrough'),
        objective = metrics.make_scorer(
            metrics.log_loss,
            needs_proba=True,
            greater_is_better=False
            )
    )

    return df, X, y, feature_columns, target_name, index, param_payload
    