# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
import pygam
from utils.ml_wrappers import LogisticGAMWrapper

from sklearn import pipeline

# hyperparam optimization
import skopt

# custom code
from utils.ml_training import ModelExperiment
from ss_defense_experiment_main_preamble import experiment_prep # abstracts away a lot of prep shared by different experiments

def main():

    df, X, y, feature_columns, target_name, index, param_payload = experiment_prep()

    # # Logistic Regression

    # experiment = ModelExperiment(
        
    #     model = LogisticRegression(max_iter=10000),
    #     space = [
    #         skopt.space.Real(0.01, 1000, "log-uniform", name='model__C')
    #         ],
    #     experiment_name = "Logistic Regression",
    #     experiment_description = f"""
    #          Logistic Regression model using fully designed and transformed inputs, all standard scaled, with no extra features added.

    #         Input Features: {feature_columns}
    #         Ouptut: {target_name}
    #         """,
    #     **param_payload
    # )
    # experiment.run_gp_inner_loop(n_calls=20)


    # GAM w/Sigmoid

    # loop to generate a sum of splines on each feature
    for i, _ in enumerate(feature_columns):
        if i == 0:
            gam_features = pygam.s(i)
        else:
            gam_features += pygam.s(i)

    experiment = ModelExperiment(
        model = pipeline.make_pipeline(LogisticGAMWrapper(gam_features, max_iter=10000, verbose=True)),
        space = [],
        experiment_name = "GAM v1",
        experiment_description = f"""
            Logistic Generalized Additive Model using fully designed and transformed inputs, all standard scaled. Default for terms 
            used is a univariate spline for each feature. No interaction terms.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_single_model_eval() # doesn't work with skopt for some reason


    # Gaussian Process Classifier

    experiment = ModelExperiment(
        
        model = GaussianProcessClassifier(),
        space = [],
        experiment_name = "Gaussian Process",
        experiment_description = f"""
             Logistic Regression model using fully designed and transformed inputs, all standard scaled, with no extra features added.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_single_model_eval() # hyperparams are automatically tuned during fitting

if __name__ == '__main__':
    main()