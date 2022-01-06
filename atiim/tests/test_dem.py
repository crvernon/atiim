import os
import tempfile
import unittest

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from atiim.dem import create_basin_dem
from atiim.package_data import SampleData


class TestDEM(unittest.TestCase):

    # expected metadata dictionary of masked raster
    COMP_METADATA = {'driver': 'GTiff',
                     'dtype': 'float32',
                     'nodata': -9999.0,
                     'width': 1114,
                     'height': 1724,
                     'count': 1,
                     'crs': CRS.from_epsg(26910),
                     'transform': Affine(0.75, 0.0, 446580.0, 0.0, -0.75, 5129541.630815)}

    def test_create_basin_dem(self):
        """Tests to ensure DEM creation is valid."""

        # load sample data
        sample_data = SampleData()

        # generate a masked raster file
        with tempfile.TemporaryDirectory() as tempdir:

            masked_raster = create_basin_dem(basin_shp=sample_data.sample_basin_shapefile,
                                             dem_file=sample_data.sample_dem,
                                             run_name='test_1',
                                             output_directory=tempdir)

            # read in raster and compare metadata against expected
            with rasterio.open(masked_raster) as src:
                self.assertEqual(TestDEM.COMP_METADATA, src.meta)


if __name__ == '__main__':
    unittest.main()
