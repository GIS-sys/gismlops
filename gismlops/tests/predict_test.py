import unittest

import numpy as np
from gismlops.infer import infer
from gismlops.train import train


class TestPredictions(unittest.TestCase):
    def testPredict(self):
        train()
        predictions = infer()
        correct = np.sum(predictions[:, 0] == predictions[:, 1])
        every = predictions.shape[0]
        PRECISION = 0.7
        self.assertTrue(correct > PRECISION * every)
