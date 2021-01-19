from uuid import uuid1
import datetime
import pickle
import json
import os
from pathlib import Path

import pandas as pd

from __init__ import *

class ModelPersistance:

    @staticmethod
    def save_model(pipeline, objective, objective_value, experiment_name, experiment_description, feature_columns, target_name):
        """
        Save a pickle file of the trained sklearn style Pipeline, 
        and record info in model registry.
        """
        model_id = str(uuid1()) # generate ID for model

        # save pickle of model (including tranformation pipeline)
        with open(f'{MODEL_REGISTRY_DIR}/{model_id}.p', 'wb') as f:
            pickle.dump(pipeline, f)

        # get_model_from_pipeline
        model = pipeline.named_steps['model'] 
            # assumes sklearn pipeline has 'model' 
            # as model
            
        # get transformation from pipeline
        transformations = pipeline.named_steps['feature_preprocessing'] 
            # assumes sklearn pipeline has 'feature_preprocessing' 
            # as only preprocessing step in inner loop

        # get details for model registry
        model_description = model.__repr__()
        timestamp_est = str(datetime.datetime.now())

        # get details for model registry
        model_description = model.__repr__()
        try:
            model_params = model.get_params()
        except: 
            # for Stan model
            model_params = 'n/a'

        if transformations is not None:
            transformation_description = str(transformations.transformers)
        else:
            # for Stan model
            transformation_description = 'n/a'

        # write to registry file
        with open(MODEL_REGISTRY_FILE, 'a') as f:
            f.write('\n')
            f.write(json.dumps(
                {
                    'id': model_id,
                    'timestamp_est': timestamp_est,
                    'model_family': model_description,
                    'params': str(model_params),
                    'feature_columns': feature_columns,
                    'transformations': transformation_description,
                    'target_name': target_name,
                    'objective_type': str(objective),
                    'objective_value': objective_value,
                    'experiment_name':experiment_name,
                    'experiment_description':experiment_description
                }
            ))
    
    @staticmethod
    def load_model_by_id(model_id):
        """
        Loads the model from pickle file. This is not a trained model,
        so you will need to run model.fit(X, y) before being able to
        make predictions.
        """
        with open(f'{MODEL_REGISTRY_DIR}/{model_id}.p', 'rb') as f:
            model = pickle.load(f)
        return model
    
    @staticmethod
    def retrieve_registry_records(sorted_by=None):
        """
        Returns regsitry records as pandas DataFrame object,
        default sorted by objective_value in ascending order.
        """
        registry = pd.read_json(MODEL_REGISTRY_FILE, lines=True)
        
        if sorted_by:
            registry = registry.sort_values(sorted_by)
        return registry
    
    @staticmethod
    def clean_registry(are_you_sure='no'):
        """
        Deletes all pickle files from registry, along with registry.jsonl file.
        """
        files = os.listdir(MODEL_REGISTRY_DIR)
        for file in files:
            if Path(file).suffix == '.p':
                os.remove(f'{MODEL_REGISTRY_DIR}/{file}')
            elif Path(file).name == 'registry.jsonl':
                os.remove(MODEL_REGISTRY_FILE)
        print('Registry cleaned.')