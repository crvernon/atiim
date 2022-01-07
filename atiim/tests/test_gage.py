import unittest

import numpy as np
import pandas as pd

import atiim.gage as gage
from atiim.package_data import SampleData


class TestGage(unittest.TestCase):
    """Tests to ensure the functionality of the gage module."""

    SAMPLE_DATA = SampleData()

    # data frame from sample data for gage csv read
    DF = gage.import_gage_data(SAMPLE_DATA.sample_gage_data_file)

    def test_import_gage_data_fields(self):
        """Ensure that expected datatypes are correct."""

        df = TestGage.DF

        # check data types for expected fields
        expected_dt_type = pd.Series({'dt': pd.to_datetime('2008-02-01 17:15:21')}).dtype
        self.assertEqual(expected_dt_type, df['date_time'].dtype)
        self.assertEqual(np.float64, df['WL_ELEV_M'].dtype)

    def test_import_gage_data_shape(self):
        """Ensure that expected shape is present."""

        df = TestGage.DF

        # confirm expected shape
        expected_shape = (3568, 4)
        self.assertEqual(expected_shape, df.shape)

    def test_import_gage_data_sort(self):
        """Ensure that data frame was sorted correctly by date_time."""

        df = TestGage.DF

        # confirm values sorted correctly by date_time
        expected_min_dt = pd.to_datetime('2008-02-01 17:15:21')
        expected_max_dt = pd.to_datetime('2008-06-29 08:15:21')
        self.assertEqual(expected_min_dt, df['date_time'].iloc[0])
        self.assertEqual(expected_max_dt, df['date_time'].iloc[-1])


    def test_import_gage_data_exception(self):
        """Ensure that expections raised when non-existent fields are passed."""

        with self.assertRaises(ValueError):

            df = gage.import_gage_data(TestGage.SAMPLE_DATA.sample_gage_data_file,
                                       date_field_name="error",
                                       time_field_name="error")


if __name__ == '__main__':
    unittest.main()
