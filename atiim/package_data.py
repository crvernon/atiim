import pkg_resources


class SampleData:
    """ATIIM sample data."""

    @property
    def sample_gage_data_file(self) -> str:
        """Retrieve the sample gage data containing a time series of water surface elevation.

        :return:                Full path with file name and extension to the sample file

        """

        return pkg_resources.resource_filename('atiim', 'data/water_level.csv')

    @property
    def sample_gage_shapefile(self) -> str:
        """Retrieve the sample gage location shapefile.

        :return:                Full path with file name and extension to the sample file

        """

        return pkg_resources.resource_filename('atiim', 'data/gage_location_1.shp')

    @property
    def sample_basin_shapefile(self) -> str:
        """Retrieve the sample basin shapefile.

        :return:                Full path with file name and extension to the sample file

        """

        return pkg_resources.resource_filename('atiim', 'data/basin_1.shp')

    @property
    def sample_dem(self) -> str:
        """Retrieve the sample DEM.

        :return:                Full path with file name and extension to the sample file

        """

        return pkg_resources.resource_filename('atiim', 'data/dem.sdat')
