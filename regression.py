import numpy as np
import pandas as pd
from scipy.optimize import minimize

class Regression:
    def __init__(self, model, objective):
        self.model = model

    def fit(self, x, y):
        result = minimize(self.model.objective, self.model.params, args=(x, y))
        self.model.params = result.x


        

