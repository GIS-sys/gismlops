import unittest

from gismlops import predict


class TestPredictions(unittest.TestCase):
    def testPredict(self):
        x, y = predict()
        self.assertEqual(x, y)
