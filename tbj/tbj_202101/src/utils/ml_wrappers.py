import pystan
import pygam
from sklearn.pipeline import Pipeline

import numpy as np
from scipy.special import expit

from abc import ABC, abstractmethod

class SkLearnModel(ABC):
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass
    

class StanModelWrapper(SkLearnModel):

    def __init__(self, model_code=None, model_file=None, fixed_effects_columns=[], **kwargs):
        if model_code and model_file:
            raise Exception('You input both a model code and a model file. Which is it? Pick one.')
        elif model_code:
            self.model = pystan.StanModel(model_code=model_code, **kwargs)
        elif model_file:
            self.model = pystan.StanModel(file=model_file, **kwargs)
        else:
            raise Exception('Must input either model_code or model_file.')

        self.fixed_effects_columns = fixed_effects_columns
        self.named_steps = {
            'model': self.model,
            'feature_preprocessing': None
        }

    def fit(self, data, iter=1000, chains=4, **kwargs):
        self.fit_results = self.model.sampling(data=data, iter=iter, chains=chains, **kwargs)
        self.params = self.fit_results.extract(permuted=True)
        #print(self.fit.stansummary())


class StanLogisticMultilevel1(StanModelWrapper):
    # implement the model itself (without the variable effects)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return self._predict_with_samples(X).mean(0) > 0.5

    def predict_proba(self, X):
        return self._predict_with_samples(X).mean(0)

    def _predict_with_samples(self, X):
        self.params = self.fit_results.extract(permuted=True)  # return a dictionary of arrays
        mu = self.params['fixed_effects']
        return expit(np.matmul(mu, X.T))

    def get_params(self):
        return self.params

    
class LogisticGAMWrapper(pygam.LogisticGAM):
    # overwrite the predict_proba method, to format like sklearn
    def __init__(self, *args, **kwargs):
        self.classes_ = np.array([0, 1])
        super(pygam.LogisticGAM, self).__init__(*args, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        preds_proba = super().predict_proba(X, **kwargs)
        sk_preds_proba = np.concatenate([ # pyGAM only outputs prob of class 1
            1-preds_proba.reshape(-1,1), 
            preds_proba.reshape(-1,1)
        ], axis=1)
        return sk_preds_proba
    
    