import numpy as np
from scripts.variable_importance import permutation_importance_array, gain_importance

class DummyModel:
    def __init__(self, coef):
        self.coef_ = np.asarray(coef)
        self.feature_gains_ = np.abs(self.coef_)
    def predict(self, X):
        z = X @ self.coef_
        return 1/(1+np.exp(-z))

def test_permutation_importance_array():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:,0] > 0).astype(int)
    model = DummyModel([1.0, 0.0])
    imp = permutation_importance_array(model, X, y, n_repeats=2, random_state=0)
    assert imp.shape == (2,2)
    assert imp[0,0] >= imp[1,0]

def test_gain_importance():
    model = DummyModel([0.1, 0.9])
    df = gain_importance(model, ["a","b"])
    assert list(df.variable) == ["b","a"]
