# Modelling
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# hyperparam optimization
import skopt

# custom code
from utils.ml_training import ModelExperiment
from ss_defense_experiment_main_preamble import experiment_prep # abstracts away a lot of prep shared by different experiments

def main():

    df, X, y, feature_columns, target_name, index, param_payload = experiment_prep()

    # # Support Vector Classifier w/ RBF Kernel

    experiment = ModelExperiment(
        
        model = SVC(
            probability=True,
            kernel='rbf'
            ),
        space = [
            skopt.space.Real(0.01, 1000, "log-uniform", name='model__C'),
            skopt.space.Real(1e-5, 1e-2, "log-uniform", name='model__tol')
            ],
        experiment_name = "SVM-RBF v1",
        experiment_description = f"""
             Support Vector Machine Classifier w/RBF kernel, using fully designed and transformed inputs, all standard scaled.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_gp_inner_loop(n_calls=20)

    # # Support Vector Classifier w/ Polynomial Kernel

    experiment = ModelExperiment(
        
        model = SVC(
            probability=True,
            kernel='poly'
            ),
        space = [
            skopt.space.Real(0.01, 1000, "log-uniform", name='model__C'),
            skopt.space.Real(1e-5, 1e-2, "log-uniform", name='model__tol'),
            skopt.space.Integer(2, 4, "uniform", name='model__degree')
            ],
        experiment_name = "SVM-Poly v1",
        experiment_description = f"""
             Support Vector Machine Classifier w/Polynomial kernel, using fully designed and transformed inputs, all standard scaled.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_gp_inner_loop(n_calls=15)

    ### Random Forest

    experiment = ModelExperiment(
        
        model = RandomForestClassifier(),
        space = [
            skopt.space.Integer(25, 2000, "log-uniform", name='model__n_estimators'),
            skopt.space.Integer(2, 20, "uniform", name='model__min_samples_split')
            ],
        experiment_name = "Random Forest",
        experiment_description = f"""
             Random Forest Classifier model using fully designed and transformed inputs, all standard scaled.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_gp_inner_loop(n_calls=20)

    ## Gradient Boosted Trees Classifier

    experiment = ModelExperiment(
        model = xgb.XGBClassifier(
            use_label_encoder=False
        ),
        space = [],
        experiment_name = "Gradient Boosted Trees",
        experiment_description = f"""
            Gradient Boosted Trees Classifier model using fully designed and transformed inputs, all standard scaled.

            Input Features: {feature_columns}
            Ouptut: {target_name}
            """,
        **param_payload
    )
    experiment.run_single_cross_validation()

if __name__ == '__main__':
    main()