import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
    
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

from .ml_utils import ModelPersistance
from .preprocessing import shortstop_global_preprocessing, shortstop_prep_inputs
from __init__ import *

def load_data_and_train_model(model_details, df, needs_proba=True):
    
    # load data and run global preprocessing
    df = pd.read_csv(f'{DATA_DIR}/shortstopdefense.csv')
    df = shortstop_global_preprocessing(df)
    
    # load model and prep inputs for those features
    model = ModelPersistance.load_model_by_id(model_details['id'])
    df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df, feature_columns=model_details['feature_columns'])
    
    model, preds, preds_proba = train_model(model, X, y, needs_proba)
        
    return model, preds, preds_prob, df, X, y, feature_columns, target_name, index

def train_model(model, X, y, needs_proba=True):

    # fit and predict
    model.fit(X,y)
    preds = model.predict(X)
    
    preds_proba = None
    if needs_proba:
        preds_proba = model.predict_proba(X)
        
    return model, preds, preds_proba

def summarize_model(all_models, current_model_name, y, preds, preds_proba, bins=100):
    # calculate brier score
    brier_score = brier_score_loss(y.values, preds_proba[:,1])
    # save to all_models payload
    all_models[current_model_name]['brier'] = brier_score
    # display summary
    print(f'{current_model_name}: {brier_score: .3f} Brier Score\n')
    plot_probability_calibration_curve(y, preds, preds_proba, current_model_name, bins=bins)
    return all_models
    
    
def plot_probability_calibration_curve(y, preds, preds_prob, name, bins=10):
    
    logloss_score = log_loss(y, preds_prob)
    
    print("\tPrecision: %1.3f" % precision_score(y, preds))
    print("\tRecall: %1.3f" % recall_score(y, preds))
    print("\tF1: %1.3f" % f1_score(y, preds))
    print("\tLog-Loss: %1.3f" % logloss_score)
    print("\tAccuracy: %1.3f\n" % accuracy_score(y, preds))

    prob_pos = preds_prob[:, 1]

    fraction_of_positives, mean_predicted_value = \
        calibration_curve(y, prob_pos, n_bins=10)
    
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="%s (%1.3f)" % (name, logloss_score))

    ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
             histtype="step", lw=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


def summarize_outs_above_average(df, y, preds_proba, min_prob_for_opportunity = 0.005):
    """
    This function prepares a dataframe that summarizes the OAA metric, given a model's probability predictions.
    """

    model_output = pd.DataFrame({
        'ss_out_probability':preds_proba,
        'observed_out':y
    })
    model_output.loc[:, 'OAA'] = model_output['observed_out'] - model_output['ss_out_probability']
    model_output.loc[:, 'opportunity'] = (model_output['ss_out_probability'] > min_prob_for_opportunity).astype(int) # over a 0.5% chance 
    
    model_output = model_output.join(df['playerid'])
    
    return model_output.reset_index()



if __name__ == '__main__':
    
    results = ModelPersistance.retrieve_registry_records(sorted_by='objective_value')
    model_details = results.iloc[0]
    model_id = model_details['id']
    model, preds, preds_prob, df, X, y, feature_columns, target_name, index = load_data_and_train_model_by_id(model_id)
    plot_probability_calibration_curve(y, preds, preds_prob, name=model_id)