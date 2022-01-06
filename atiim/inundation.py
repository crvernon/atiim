import os
import logging
import warnings
from typing import Union

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio.mask
import shapely.speedups
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from joblib import delayed, Parallel

from .gage import process_gage_data
from .dem import create_basin_dem

# enable shapely speedups for topology operations
shapely.speedups.enable()


# TODO:  remove ignore warning generated for a future shapely depreciation in rasterio.features.shapes
warnings.filterwarnings('ignore')


def process_slice(arr: np.ndarray,
                  upper_elev: float,
                  gage_gdf: gpd.GeoDataFrame,
                  water_elev_freq: dict,
                  run_name: str,
                  hour_interval: float,
                  transform: Affine,
                  target_crs: CRS,
                  write_shapefile: bool = True,
                  output_directory: Union[str, None] = None) -> gpd.GeoDataFrame:
    """Create a water level polygon shapefile containing a single feature that represents
    the grid cells of an input DEM that are less than or equal to an upper elevation level.

    :param arr:                     2D array from raster band read
    :type arr:                      np.ndarray

    :param upper_elev:              Elevation value for the upper bound of the elevation interval
    :type upper_elev:               float

    :param gage_gdf:                GeoDataFrame of the gage location point
    :type gage_gdf:                 gpd.GeoDataFrame

    :param water_elev_freq:         Dictionary of water elevation frequency {elev:  frequency}
    :type water_elev_freq:          dict

    :param run_name:                Name of run, all lowercase and only underscore separated.
    :type run_name:                 str

    :param hour_interval:           Time step of inundation extent.  Either 1.0 or 0.5.
    :type hour_interval:            float

    :param transform:               Trasformation object from the rasterio source raster
    :type transform:                Affine

    :param target_crs:              Coordinate reference system (CRS) object from the rasterio source raster to
                                    be used in projecting the water level polygons.
    :type target_crs:               CRS

    :param write_shapefile:         Optional.  Choice to write the GeoDataFrame water level polygon as a shapefile.
                                    Set output directory if True.  Default is True.

    :type write_shapefile:          bool

    :param output_directory:        Full path to a write-enabled directory to write output files to if write_shapefile
                                    is set to True
    :type output_directory:         str

    :return:                        A geopandas data frame of a polygon intersecting the gage point location for
                                    the target elevation interval.

    """

    # TODO:  account for different units
    if hour_interval not in (1.0, 0.5):
        msg = f"The hour interval of '{hour_interval}' is not currently supported.  Please use either 1.0 or 0.5 (half hour)."
        raise AssertionError(msg)

    # generate a feature id from the elevation value
    feature_id = int(upper_elev * 100)

    # create every value greater than or equal to the upper elevation to 1, others to 0
    arx = np.where(arr <= upper_elev, 1, 0).astype(np.int16)

    # build each feature based on the extracted grid cells from the array
    results = list(
        {'properties': {'raster_val': val}, 'geometry': shp}
        for index, (shp, val) in enumerate(
            shapes(arx, mask=None, transform=transform))
    )

    # list of geometries
    geoms = list(results)

    # build geopandas dataframe from geometries
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=target_crs)

    # only keep the ones
    gdf = gdf.loc[gdf['raster_val'] == 1]

    # only keep the polygon intersecting the gage
    gdf['valid'] = gdf.intersects(gage_gdf.geometry.values[0])
    gdf = gdf.loc[gdf['valid']].copy()

    # ensure at least one polygon intersects the gage
    if gdf.shape[0] == 0:
        msg = "Gage location point not aligned with valid elevation in DEM.  Relocate gage location point to fall within valid elevation."
        raise AssertionError(msg)

    # dissolve into a single polygon
    gdf = gdf.dissolve('raster_val')
    gdf.reset_index(inplace=True)

    # add fields
    gdf['id'] = feature_id
    gdf['frequency'] = water_elev_freq[round(upper_elev, 1)]
    gdf['elevation'] = upper_elev
    gdf['area'] = gdf.geometry.area
    gdf['hectares'] = gdf['area'] * 0.0001
    gdf['perimeter'] = gdf.geometry.length
    gdf['hect_hours'] = (gdf['frequency'] / hour_interval) * gdf['hectares']
    gdf['run_name'] = run_name

    # drop unneeded fields
    gdf.drop(columns=['raster_val', 'valid'], inplace=True)

    # write to file if desired
    if write_shapefile:

        if output_directory is None:
            msg = 'Please pass a value for output_directory if choosing to write shapefile outputs.'
            raise AssertionError(msg)

        out_file = os.path.join(output_directory, f'wl_{feature_id}_{run_name}.shp')
        gdf.to_file(out_file)

    return gdf


def simulate_inundation(dem_file: str,
                        basin_shp: str,
                        gage_shp: str,
                        gage_data_file: str,
                        run_name: str,
                        output_directory: Union[str, None] = None,
                        write_csv: bool = True,
                        elevation_interval: float = 0.1,
                        hour_interval: float = 1.0,
                        n_jobs: int = 1,
                        verbose: bool = False):
    """Worker function to simulate inundation over a DEM using a stable gage location and accompanying water level
    time series data.

    :param dem_file:                Full path with file name and extension to the input DEM raster file.
    :type dem_file:                 str

    :param basin_shp:               Full path with file name and extension to the target basin shapefile
    :type basin_shp:                str

    :param gage_shp:                Full path with file name and extension to the target gage shapefile
    :type gage_shp:                 str

    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param output_directory:        Full path to a write-enabled directory to write output files to
    :type output_directory:         str

    :param run_name:                Name of run, all lowercase and only underscore separated.
    :type run_name:                 str

    :param write_csv:               Choice to write a CSV file of inundation metric outputs
    :type write_csv:                bool

    :param elevation_interval:      Step for elevation to be processed.
    :type elevation_interval:       float

    :param hour_interval:           Time step of inundation extent.  Either 1.0 or 0.5.
    :type hour_interval:            float

    :param n_jobs:                  The maximum number of concurrently running jobs, such as the number of Python
                                    worker processes when backend=”multiprocessing” or the size of the thread-pool
                                    when backend=”threading”. If -1 all CPUs are used. If 1 is given, no parallel
                                    computing code is used at all, which is useful for debugging. For n_jobs
                                    below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all
                                    CPUs but one are used. None is a marker for ‘unset’ that will be interpreted
                                    as n_jobs=1 (sequential execution) unless the call is performed under
                                    a parallel_backend context manager that sets another value for n_jobs.
    :type n_jobs:                   int

    :param verbose:                 Choice to log verbosely
    :type verbose:                  bool

    """

    # process gage data file
    min_gage_elev, max_gage_elev, water_elev_freq = process_gage_data(gage_data_file, verbose=verbose)

    # read in gage shapefile to a geodataframe
    gage_gdf = gpd.read_file(gage_shp)

    with rasterio.Env():

        # clip the input DEM to a target basin contributing area
        masked_dem_file = create_basin_dem(basin_shp, dem_file, run_name, output_directory)

        with rasterio.open(masked_dem_file) as src:

            # read the raster band into a number array
            arr = src.read(1)

            # convert the raster nodata value to numpy nan
            arr[arr == src.nodata] = np.nan

            raster_min = float(np.nanmin(arr))
            raster_max = float(np.nanmax(arr))

            # use the minimum bounding elevation e.g., the max of min available
            elev_min = max([min_gage_elev, raster_min])

            # use the maximum bounding elevation e.g., the min of max available
            elev_max = min([max_gage_elev, raster_max])

            # construct elevation upper bounds to process for each slice
            elev_slices = np.arange(elev_min, elev_max + elevation_interval, elevation_interval)

            # process all elevation slices in parallel
            feature_list = Parallel(n_jobs=n_jobs)(
                delayed(process_slice)(arr=arr,
                                       upper_elev=upper_elev,
                                       output_directory=output_directory,
                                       gage_gdf=gage_gdf,
                                       water_elev_freq=water_elev_freq,
                                       run_name=run_name,
                                       hour_interval=hour_interval,
                                       transform=src.transform,
                                       target_crs=src.crs)
                for upper_elev in elev_slices)

            # concatenate individual GeoDataFrames into a single frame
            result_df = pd.concat(feature_list)

            # drop geometry
            result_df.drop(columns=['geometry'], inplace=True)

            # write data frame to file removing geometry if so desired
            if write_csv:

                if output_directory is None:
                    msg = 'Please pass a value for output_directory if choosing to write shapefile outputs.'
                    raise AssertionError(msg)

                out_file = os.path.join(output_directory, f'inundation_metrics_{run_name}.csv')
                result_df.to_csv(out_file, index=False)

            if verbose:
                logging.info(f"Minimum DEM Elevation:  {round(raster_min, 2)}")
                logging.info(f"Maximum DEM Elevation:  {round(raster_max, 2)}")
                logging.info(f"Bounded DEM Elevation:  {round(min(elev_slices), 2)}")
                logging.info(f"Bounded DEM Elevation:  {round(max(elev_slices), 2)}")

            return pd.DataFrame(result_df)
