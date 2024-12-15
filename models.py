import numpy as np

__all__ = ["LinearModel", "LinearGaussianModel"]

class BaseModel:
    def __init__(self, initial_params, criterion):
        self.params = (
            initial_params
            if isinstance(initial_params, np.ndarray)
            else np.array(initial_params)
        )
        self.criterion = criterion

    def __call__(self, x, params=None):
        raise NotImplementedError

    def objective(self, params, x, y):
        predictions = self(x, params)
        return self.criterion(predictions, y)


class LinearModel(BaseModel):
    def __call__(self, x, params=None):
        if params is None:
            params = self.params
        return (params[0] + params[np.newaxis, 1:] @ x.T).squeeze()

class LinearGaussianModel(BaseModel):
    def __call__(self, x, params=None, return_std=False):
        if params is None:
            params = self.params
        mu = params[0] + params[np.newaxis, 1:-1] @ x.T
        if return_std:
            return mu.squeeze(), params[-1]
        return mu.squeeze()

    def objective(self, params, x, y):
        mu, std = self(x, params, return_std=True)
        if std <= 0:
            return np.inf
        NLL = np.log(std)*x.shape[0] + 0.5 * np.sum(((y - mu) / std) ** 2)
        return NLL