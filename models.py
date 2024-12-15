import numpy as np


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
