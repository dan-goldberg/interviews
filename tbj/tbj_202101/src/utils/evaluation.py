import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from .ml_utils import ModelPersistance
from .preprocessing import shortstop_global_preprocessing, shortstop_prep_inputs
from __init__ import *

def load_data_and_train_model(model_details, needs_proba=True):
    
    # load data and run global preprocessing
    df = pd.read_csv(f'{DATA_DIR}/shortstopdefense.csv')
    df = shortstop_global_preprocessing(df)
    
    # load model and prep inputs for those features
    model = ModelPersistance.load_model_by_id(model_details['id'])
    df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df, feature_columns=model_details['feature_columns'])
    
    # fit and predict
    model.fit(X,y)
    preds = model.predict(X)
    
    preds_prob = None
    if needs_proba:
        preds_prob = model.predict_proba(X)
        
    return model, preds, preds_prob, df, X, y, feature_columns, target_name, index




if __name__ == '__main__':
    
    results = ModelPersistance.retrieve_registry_records(sorted_by='objective_value')
    model_details = results.iloc[0]
    model_id = model_details['id']
    model, preds, preds_prob, df, X, y, feature_columns, target_name, index = load_data_and_train_model_by_id(model_id)
    plot_probability_calibration_curve(y, preds, preds_prob, name=model_id)