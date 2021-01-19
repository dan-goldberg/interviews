import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# sklearn preprocessing
from sklearn.preprocessing import OrdinalEncoder

# Modelling

from utils.ml_wrappers import StanLogisticMultilevel1

# model evaluation 
from sklearn import metrics #log_loss, accuracy_score
from sklearn import model_selection # cross_val_score

# hyperparam optimization
import skopt

# custom code
from utils.ml_training import ModelPersistance
from ss_defense_experiment_main_preamble import experiment_prep # abstracts away a lot of prep shared by different experiments

from __init__ import DATA_DIR

def main():

    df, X, y, feature_columns, target_name, index, param_payload = experiment_prep()

    model = StanLogisticMultilevel1(model_file='stan_configs/logistic_multilevel_001.stan', fixed_effects_columns=feature_columns)

    experiment_details = dict(
        model = model,
        space = None,
        experiment_name = "Multilevel Logistic Regression v1",
        experiment_description = f"""
            Hierarchical model same as vanilla logistic regression plus a random intercept 
            for the shortstop identity, trained with NUTS MCMC sampler in Stan. The hope
            is to control for confounder of player identity and ground ball profile (difficulty),
            i.e. the pitchers on the player's team give up hard ground balls more often. 

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )

    # reindex player id
    oe = OrdinalEncoder(dtype=int)
    df.loc[:, 'playerid_cat'] = oe.fit_transform(df[['playerid']]) 
    levels = df['playerid_cat'].values + 1 # reindex with 1-index
    num_levels = np.unique(levels).shape[0]

    column_transformer = param_payload['feature_preprocessing']
    X_t = column_transformer.fit_transform(X[feature_columns])

    data = {
        'num_samples': X_t.shape[0],
        'num_features': X_t.shape[1],
        'feature1': X_t[:,0],
        'feature2': X_t[:,1],
        'feature3': X_t[:,2],
        'feature4': X_t[:,3],
        'feature5': X_t[:,4],
        'feature6': X_t[:,5],
        'feature7': X_t[:,6],
        'feature8': X_t[:,7],
        'feature9': X_t[:,8],
        'feature10': X_t[:,9],
        'num_levels': num_levels,
        'level': levels,
        'labels':y
    }

    # fit model
    model.fit(data, control={'max_treedepth': 15})

    # evaluate model
    params = model.params  # return a dictionary of arrays
    bias = params['bias'].mean(axis=0)
    slope1 = params['slope1'].mean(axis=0)
    slope2 = params['slope2'].mean(axis=0)
    slope3 = params['slope3'].mean(axis=0)
    slope4 = params['slope4'].mean(axis=0)
    slope5 = params['slope5'].mean(axis=0)
    slope6 = params['slope6'].mean(axis=0)
    slope7 = params['slope7'].mean(axis=0)
    slope8 = params['slope8'].mean(axis=0)
    slope9 = params['slope9'].mean(axis=0)
    slope10 = params['slope10'].mean(axis=0)
    level_param = params['shortstop_effect'].mean(axis=0)

    # get predictions
    preds_proba = bias \
        + slope1*X_t[:,0] \
        + slope2*X_t[:,1] \
        + slope3*X_t[:,2] \
        + slope4*X_t[:,3] \
        + slope5*X_t[:,4] \
        + slope6*X_t[:,5] \
        + slope7*X_t[:,6] \
        + slope8*X_t[:,7] \
        + slope9*X_t[:,8] \
        + slope10*X_t[:,9] \
        + [level_param[l-1] for l in levels]

    preds = (expit(preds_proba) > 0.5).astype(int)

    result_logloss = metrics.log_loss(y, preds_proba)
    result_accuracy = metrics.accuracy_score(y, preds)

    print(f'GLMM Logloss: {result_logloss}')
    print(f'GLMM Accuracy: {result_accuracy}')

    ModelPersistance.save_model(
        model, 
        metrics.log_loss, 
        result_logloss,
        experiment_details['experiment_name'],
        experiment_details['experiment_description'],
        feature_columns + ['playerid_cat+1'],
        target_name
        )

    return model

if __name__ == '__main__':
    main()