import unittest

from gismlops import learn, predict


class TestPredictions(unittest.TestCase):
    def testPredict(self):
        learn()
        x, y = predict()
        self.assertEqual(x, y)
