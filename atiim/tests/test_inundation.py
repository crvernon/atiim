import unittest
import pkg_resources

import pandas as pd

import atiim.inundation as ind
from atiim.package_data import SampleData


class TestInundation(unittest.TestCase):
    """Tests for the inundation module."""

    SAMPLE_DATA = SampleData()

    # data frame for comparison of output
    COMP_FILE = pkg_resources.resource_filename('atiim', 'tests/data/test_simulate_inundation_output.zip')
    COMP_DF = pd.read_csv(COMP_FILE, index_col='id')

    def test_simulate_inundation(self):
        """Test full data frame equality with expected."""

        df = ind.simulate_inundation(dem_file=TestInundation.SAMPLE_DATA.sample_dem,
                                     basin_shp=TestInundation.SAMPLE_DATA.sample_basin_shapefile,
                                     gage_shp=TestInundation.SAMPLE_DATA.sample_gage_shapefile,
                                     gage_data_file=TestInundation.SAMPLE_DATA.sample_gage_data_file,
                                     run_name='test')

        df.set_index('id', inplace=True)

        pd.testing.assert_frame_equal(TestInundation.COMP_DF, df)


if __name__ == '__main__':
    unittest.main()
