import numpy as np


class NormPQN:
    def __init__(self, raw_data):
        self.data = raw_data

    def normalize(self):
        data = self.data

        data = data.apply(np.float64)
        assert not data.isnull().any().any()

        reference_spectrum = data.median(axis=0)
        scaling_factor = (reference_spectrum / data).median(axis=1)
        data_norm = (data.T * scaling_factor).T

        self.data_normalized = data_norm

        print("Done with normalization")
