import unittest

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

    SAMPLE_DATA = SampleData()

    def test_create_basin_dem_vaid(self):
        """Tests to ensure DEM creation is valid."""

        arr, meta = create_basin_dem(basin_shp=TestDEM.SAMPLE_DATA.sample_basin_shapefile,
                                     dem_file=TestDEM.SAMPLE_DATA.sample_dem,
                                     run_name='test_1',
                                     write_raster=False)

        # read in raster and compare metadata against expected
        self.assertEqual(TestDEM.COMP_METADATA, meta)

    def test_create_basin_dem_exception(self):
        """Tests to ensure function raises expected exceptions."""

        # raise assertion error when not passing output directory when write_raster is True
        with self.assertRaises(AssertionError):

            arr, meta = create_basin_dem(basin_shp=TestDEM.SAMPLE_DATA.sample_basin_shapefile,
                                         dem_file=TestDEM.SAMPLE_DATA.sample_dem,
                                         run_name='test_1',
                                         write_raster=True,
                                         output_directory=None)


if __name__ == '__main__':
    unittest.main()
