import unittest

import numpy as np

import atiim.hypsometric as hyp
from atiim.package_data import SampleData


class TestHysometric(unittest.TestCase):
    """Tests to ensure the functionality of the hypsometric module."""

    SAMPLE_DATA = SampleData()

    def test_hypsometric_curve(self):
        """Ensure the hypsometric curve function performs correctly."""

        df = hyp.hypsometric_curve(TestHysometric.SAMPLE_DATA.sample_dem,
                                   elevation_interval=1.0,
                                   min_elevation=None,
                                   max_elevation=None)

        # test shape
        self.assertEqual((21, 3), df.shape)

        # check all field data types
        self.assertEqual(True, np.all([i == np.float64 for i in df.dtypes]))

        # ensure the file summary is correct
        self.assertEqual(df.min(axis=1).min().round(1), 0.0)
        self.assertEqual(df.max(axis=1).max().round(1), 516908.2),
        self.assertEqual(df.std(axis=1).std().round(1), 81706.0),
        self.assertEqual(df.mean(axis=1).mean().round(1), 59154.9)


if __name__ == '__main__':
    unittest.main()
