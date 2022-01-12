import unittest

import numpy as np
import pandas as pd

from atiim.inflection import calculate_bankfull_elevation


class TestInflection(unittest.TestCase):
    """Tests to confirm inflection code efficacy."""

    TEST_DF = pd.DataFrame({'elevation': np.arange(0.1, 1.1, 0.1),
                            'area': [1.0, 10.0, 12.4, 10.2, 14.1, 15.7, 20.4, 18.2, 40.4, 42.7]})

    def test_bankfull_elevation(self):
        """Ensure the first inflection point is captured."""

        elev, area = calculate_bankfull_elevation(TestInflection.TEST_DF)

        self.assertEqual((0.3, 12.4), (round(elev, 1), area))


if __name__ == '__main__':
    unittest.main()
