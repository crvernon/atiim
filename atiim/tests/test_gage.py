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

    def test_process_gage_data(self):
        """Ensure..."""

        min_wtr_elev, max_wtr_elev, d_freq = gage.process_gage_data(TestGage.SAMPLE_DATA.sample_gage_data_file)

        # expected dictionary of elevation frequencies
        comp_dict = {1.1: 523, 1.2: 462, 1.3: 231, 1.4: 177, 2.5: 172, 2.3: 162, 2.0: 158, 1.6: 155, 2.2: 154,
                     1.5: 151, 1.9: 151, 2.1: 136, 1.8: 132, 2.4: 132, 1.7: 131, 2.6: 113, 2.7: 108, 2.8: 101,
                     2.9: 76, 3.0: 59, 3.1: 44, 3.2: 17, 3.3: 11, 3.4: 4, 3.5: 3, 3.6: 3, 1.0: 2}

        self.assertEqual(1.0, min_wtr_elev)
        self.assertEqual(3.6, max_wtr_elev)
        self.assertEqual(comp_dict, d_freq)


if __name__ == '__main__':
    unittest.main()
