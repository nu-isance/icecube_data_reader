import numpy as np


class DummyPDF:
    def __init__(self):
        pass

    def pdf(self, x):
        if isinstance(x, np.ndarray):
            return np.zeros_like(x)
        else:
            return 0.0

    def cdf(self, x):
        if isinstance(x, np.ndarray):
            return np.zeros_like(x)
        else:
            return 0.0

    def rvs(self, size=1, random_state=42):
        raise NotImplementedError("Dummies cannot be sampled from!")
