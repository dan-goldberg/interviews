# all imports are handled in __init__.py

# Global Preprocessing to generate global dataset
df = pd.read_csv('../data/shortstopdefense.csv')
df = shortstop_global_preprocessing(df)
df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df)

feature_columns = [
    'launch_speed',
    'launch_vert_ang',
    'landing_location_radius', # maybe
    'hang_time', # maybe
    'player_time_to_point_of_interest',
    'ball_time_to_point_of_interest', # maybe this should be eliminated b/c is superfluous w/ player_time_to_point_of_interest and player_time_minus_ball_time
    'player_time_minus_ball_time', # always positive, lots of 0s for plays made
    'player_angle_from_interception_point_to_base_of_interest',
    'player_distance_from_interception_point_to_base_of_interest'
]

log_transform_feature_columns = [
    'hang_time',
]
standard_scalar_feature_columns = [
    'launch_speed',
    'launch_vert_ang',
    'landing_location_radius',
    'hang_time',
    'player_time_to_point_of_interest',
    'ball_time_to_point_of_interest',
    'player_time_minus_ball_time',
    'player_angle_from_interception_point_to_base_of_interest',
    'player_distance_from_interception_point_to_base_of_interest'
]

experiment = ModelExperiment(

    X = X,
    y = y,
    target_name = target_name,
    
    model = xgb.sklearn.XGBClassifier(),
    
    feature_preprocessing = ColumnTransformer([
            ('log_transform', FunctionTransformer(np.log, np.exp), log_transform_feature_columns),
            ('standard_scalar', StandardScaler(), standard_scalar_feature_columns)
        ], remainder='passthrough'),
    
    space = [
        skopt.space.Integer(10, 50, "uniform", name='model__max_depth'),
        skopt.space.Real(0.01, 1.5, "log-uniform", name='model__eta')
        ],
    
    objective = metrics.make_scorer(
        metrics.accuracy_score,
        greater_is_better=True
    ),
    
    experiment_description = f"""
        Basic experiment for shortstop defense with XGBoost classifier and accuracy scoring.

        Input Features: {feature_columns}
        Ouptut: {target_name}
        """
)