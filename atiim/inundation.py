import os

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.mask
import shapely.speedups

from rasterio.plot import show
from rasterio.features import shapes
from shapely.geometry import shape
from joblib import delayed, Parallel
from scipy.ndimage import gaussian_filter1d


def calculate_bankfull_elevation(df: pd.DataFrame,
                                 smooth_data: bool = False,
                                 smooth_sigma: int = 100,
                                 area_field_name: str = 'area',
                                 elevation_field_name: str = 'elevation') -> float:
    """Calculate the bankfull elevation point, or the first point of inflection, using the index of the
    change in sign of the second derivatives of the accumulating area with increasing elevation.  If the
    input data is noisy, you may choose to smooth the data using a Gaussian filter.

    :param df:                      Desingned for output from the atiim.simulate_inundation() function.  Though can
                                    be used with any data frame having an area and elevation field with data.
    :type df:                       pd.DataFrame

    :param smooth_data:             Optional.  Smooth noisy data using a Gaussian filter.  Use in combination with
                                    the smooth_sigma setting.
    :type smooth_data:              bool

    :param smooth_sigma:            Optional.  Standard deviation for Gaussian kernel.  Use when smooth_data is set
                                    to True.
    :type smooth_sigma:             int

    :param area_field_name:         Optional.  Name of area field in data frame.  Default:  'area'
    :type area_field_name:          str

    :param elevation_field_name:    Optional.  Name of elevation field in data frame.  Default 'elevation'
    :type elevation_field_name:     str

    :return:                        Bankfull elevation value.  First inflection point.

    """

    # smooth data using a Gaussian filter to remove noise if desired
    if smooth_data:
        area_data = gaussian_filter1d(df[area_field_name], sigma=smooth_sigma)
    else:
        area_data = df[area_field_name]

    # calculate the second derivatives
    second_derivatives = np.gradient(np.gradient(area_data))

    # get the index locations in the second derivative plot representing the sign change (a.k.a., inflection points)
    inflection_indices = np.where(np.diff(np.sign(second_derivatives)))[0]

    # drop the first value in the series if it shows up as an inflection point
    inflection_indices = inflection_indices[inflection_indices > 0]

    # bankfull elevation is determined by the first non-zero index inflection point
    bankfull_elevation = df[elevation_field_name].values[inflection_indices[0]]

    return bankfull_elevation


def process_gage_data(gage_data_file: str,
                      data_field_name: str = 'DATE',
                      time_field_name: str = 'TIME',
                      elevation_field_name: str = 'WL_ELEV_M'):
    """Process gage data tabular file.
    
    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param data_field_name:         Name of date field in file
    :type data_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

    :param elevation_field_name:    Name of elevation field in file
    :type elevation_field_name:     str

    :returns:                       [0] minumum water elevation in file
                                    [1] maximum water elevation in file
                                    [2] dictionary of water elevation frequency {elev:  frequency}
    """

    df = pd.read_csv(gage_data_file)

    print(f"Total Time Steps:  {df.shape[0]}")

    # convert date and time strings to a pandas datetime type
    df['date_time'] = pd.to_datetime(df[data_field_name] + ' ' + df[time_field_name], infer_datetime_format=True)

    # calculate the number of days in the file
    n_days = (df['date_time'].max() - df['date_time'].min()).days

    print(f"Days Verification:  {n_days}")

    # sort df by date_time
    df.sort_values(by=['date_time'], inplace=True)

    min_wtr_elev = df[elevation_field_name].min()
    max_wtr_elev = df[elevation_field_name].max()
    d_freq = df[elevation_field_name].value_counts().to_dict()

    return min_wtr_elev, max_wtr_elev, d_freq


def create_basin_dem(basin_shp: str,
                     dem_file: str,
                     output_directory: str,
                     run_name: str):
    """Mask the input DEM using a basin geometry representative of the contributing area.

    :param basin_shp:               Full path with file name and extension to the target basin shapefile
    :type basin_shp:                str

    :param dem_file:                Full path with file name and extension to the input DEM raster file.
    :type dem_file:                 str

    :param output_directory:        Full path to a write-enabled directory to write output files to
    :type output_directory:         str

    :param run_name:                Name of run, all lowercase and only underscore separated.
    :type run_name:                 str

    :return:                        Full path with file name and extension to the masked DEM raster file

    """

    # dissolve target basin geometries
    basin_geom = gpd.read_file(basin_shp).dissolve().geometry.values[0]

    with rasterio.open(dem_file) as src:
        if src.crs is None:
            print("Warning:  Input DEM raster does not have a defined coordinate reference system.")

        # apply basin geometry as a mask
        out_image, out_transform = rasterio.mask.mask(src, basin_geom, crop=True)

        # update the raster metadata with newly cropped extent
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # write outputs
        output_file = os.path.join(output_directory, f"dem_masked_{run_name}.tif")
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        return output_file
