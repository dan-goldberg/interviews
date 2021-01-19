import datetime

import numpy as np

# sklearn pipelines
from sklearn import pipeline #Pipeline
# model evaluation 
from sklearn import model_selection # cross_val_score
# hyperparam optimization
import skopt
    #skopt.space.Real, skopt.space.Real
    #skopt.utils.use_named_args

from .ml_utils import ModelPersistance

class ModelExperiment:
    """
    Class containing attributes of the sklearn style model class and skopt seach space
    
    param: model: 
        # instantiated scikit-learn style model class, 
        
        i.e. 
        sklearn.svm.SVR
        xgboost.XGBRegressor

    param: feature_preprocessing:
        # scikit-learn style Pipeline

        i.e. 
        ColumnTransformer([
            ('standard_scalar', StandardScaler(), standard_scalar_feature_columns)
        ], remainder='passthrough')
    
    param: objective:
        # scoring objective using sklearn.metrics.make_scorer 
        # on sklearn metric 
        
        i.e. 
        metrics.make_scorer(metrics.mean_absolute_error)
    
    param: space:
        # The list of hyper-parameters we want to optimize. For each one we define the
        # bounds, the corresponding scikit-learn parameter name, as well as how to
        # sample values from that dimension (`'log-uniform'` for the learning rate)

        i.e.
        space  = [
            skopt.space.Real(0.01, 100, "log-uniform", name='model__C') 
                # prepend with 'model__' due to using an sklearn 
                # pipeline with model step named 'model'
        ]
    """
    
    def __init__(self, 
            X,
            y,
            model, 
            feature_preprocessing,
            objective, 
            space,
            experiment_name,
            experiment_description,
            feature_columns,
            target_name,
            n_cross_val = 5
            ):
        """This class runs an experiment training a particular kind of sklearn model
        over a space of hyperparameters in order to find the optimimal hyperparameter 
        settings, using bayesian optimization with a gaussian process estimator, and 
        k-fold cross validation in the inner loop.

        All model configs over the course of this experiment are saved and registered
        for easy load and use after the fact. 

        Args:
            X (numpy.array or pandas.DataFrame): Input matrix of samples x features.
            y (numpy.array or pandas.DataFrame): Output variable.
            model (sklearn model): Model used to fit and predict on input.
            feature_preprocessing (sklearn ColumnTransformer): The column 
                preprocessing done before training, within inner loop 
                of cross validation.
            objective (sklearn scorer): model evaluation metric 
                being optimized by the gaussian optimization scheme.
            space (skopt.Space): skopt Space defining the hyperparameter 
                space to optimize over.
            experiment_description (str): String description of this experiment.
            target (str): Column name for prediction variable.
            n_cross_val (int, optional): Number of cross validation rounds. Defaults to 5.
        """
        self.X = X
        self.y = y
        self.pipeline = self._make_pipeline(model, feature_preprocessing)
        self.objective = objective
        self.space = space
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        self.feature_columns = feature_columns
        self.target_name = target_name
        self.n_cross_val = n_cross_val

    def _make_pipeline(self, model_base, feature_preprocessing):
        return pipeline.Pipeline([
            ('feature_preprocessing', feature_preprocessing),
            ('model', model_base)
        ])

    def _make_gp_inner_loop(self, pipeline, X, y):
        """This internal method defines and return the function
        for the gaussian optimization inner loop, gp_minimize_objective().

        Returns:
            function: gp_minimize_objective; this function is the function
            to be passed into skopt.gp_minimize which is the function that 
            the gaussian optimization is trying to minimize
        """

        scoring = self.objective

        @skopt.utils.use_named_args(self.space)
        def gp_minimize_objective(**params):
            """"
            Code extended from https://scikit-optimize.github.io/stable/auto_examples/hyperparameter-optimization.html#sphx-glr-auto-examples-hyperparameter-optimization-py
            
            This decorator allows your objective function to receive a the parameters as
            keyword arguments. This is particularly convenient when you want to set
            scikit-learn estimator parameters
            """
            pipeline.set_params(**params)
            result = self._run_cross_validation_scoring(
                    pipeline,
                    X, 
                    y,
                    scoring = scoring,
                    params = params
            ) # should be something to minimize
            return result

        return gp_minimize_objective

    def _run_cross_validation_scoring(self, pipeline, X, y, scoring, params=None):
        score = model_selection.cross_val_score(
                    pipeline,
                    X, 
                    y, 
                    cv=self.n_cross_val, 
                    n_jobs=-1,
                    scoring=scoring
                )
        result = -np.mean(score)
        self._save_model(pipeline, scoring, result, params)
        return result
    
    def _save_model(self, pipeline, scoring, result, params):
        print(datetime.datetime.now(), f'result: {result}', f'params: {params}')
        ModelPersistance.save_model(pipeline, scoring, result, self.experiment_name, self.experiment_description, self.feature_columns, self.target_name)

    def run_gp_inner_loop(self, n_calls=10, **kwargs):
        gp_minimize_objective = self._make_gp_inner_loop(self.pipeline, self.X, self.y)
        res_gp = skopt.gp_minimize(gp_minimize_objective, self.space, n_calls=n_calls, random_state=0, **kwargs)
        return res_gp

    def run_single_cross_validation(self):
        return self._run_cross_validation_scoring(self.pipeline, self.X, self.y, self.objective)
    
    def run_single_model_eval(self):
        self.pipeline.fit(self.X, self.y)
        result = -self.objective(self.pipeline, self.X, self.y)
        self._save_model(self.pipeline, self.objective, result, params='No BayesOpt')
        return result
