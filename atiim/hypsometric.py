import rasterio
import numpy as np
import pandas as pd

from typing import Union


def hypsometric_curve(dem_file: str,
                      elevation_interval: float = 0.1,
                      min_elevation: Union[float, None] = None,
                      max_elevation: Union[float, None] = None) -> pd.DataFrame:
    """Calculate a hypsometric curve as an elevation-area relationship assessment metric
    of the landform shape at a site.  Provides basic metric of opportunity for inundation and
    habitat opportunity.

    :param dem_file:                Full path with file name and extension to the input digital elevation model
                                    raster
    :type dem_file:                 str

    :param elevation_interval:      Elevation sample spacing in the units of the input DEM
    :type elevation_interval:       float

    :param min_elevation:           Optional.  Minimum elevation to sample from.  Default is to use the minimum
                                    elevation of the raster as a starting point.
    :type min_elevation:            float

    :param max_elevation:           Optional.  Maximum elevation to sample from.  Default is to use the maximum
                                    elevation of the raster as a starting point.
    :type max_elevation:            float

    :return:                        Pandas DataFrame of elevation, area at or above the target elevation, and
                                    percent area at or above the target elevation for each elevation interval

    """

    # create a dictionary to hold results
    result_dict = {'dem_elevation': [], 'dem_area_at_elevation': []}

    with rasterio.open(dem_file) as src:

        # read the raster band into a number array
        arr = src.read(1)

        # convert the raster nodata value to numpy nan
        arr[arr == src.nodata] = np.nan

        # grid cell resolution
        grid_cell_area = np.abs(src.transform[0] * src.transform[4])

        # set minimum and maximum elevation value; use raster determined values by default
        if min_elevation is None:
            min_elevation = np.nanmin(arr)

        if max_elevation is None:
            max_elevation = np.nanmax(arr)

        # create elevation intervals to process
        elev_slices = np.arange(min_elevation, max_elevation + elevation_interval, elevation_interval)

        # calculate each area at elevation intervals
        for i in elev_slices:
            result_dict['dem_elevation'].append(i)
            result_dict['dem_area_at_elevation'].append(np.where(arr >= i)[0].shape[0] * grid_cell_area)

        # convert results to data frame
        df = pd.DataFrame(result_dict)

        # calculate the total area
        total_area = df['dem_area_at_elevation'].max()

        # calculate percent area per elevation slice
        df['dem_percent_area'] = df['dem_area_at_elevation'] / total_area

        return df
