import pystan
import numpy as np
from scipy.special import expit

from abc import ABC, abstractmethod

class StanModelWrapper(ABC):

    def __init__(self, model_code=None, model_file=None, fixed_effects_columns=[]):
        if model_code and model_file:
            raise Exception('You input both a model code and a model file. Which is it? Pick one.')
        elif model_code:
            self.model = pystan.StanModel(model_code=model_code)
        elif model_file:
            self.model = pystan.StanModel(file=model_file)
        else:
            raise Exception('Must input either model_code or model_file.')

        self.fixed_effects_columns = fixed_effects_columns

    def fit(self, data, iter=1000, chains=4, **kwargs):
        self.fit_results = self.model.sampling(data=data, iter=iter, chains=chains, **kwargs)
        #print(self.fit.stansummary())

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass



class StanLogisticMultilevel1(StanModelWrapper):
    # implement the model itself (without the variable effects)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return self._predict_with_samples(X).mean(0) > 0.5

    def predict_proba(self, X):
        return self._predict_with_samples(X).mean(0)

    def _predict_with_samples(self, X):
        params = self.fit_results.extract(permuted=True)  # return a dictionary of arrays
        mu = params['fixed_effects']
        return expit(np.matmul(mu, X[self.fixed_effects_columns].values.T))
