import os
import logging
import tempfile
from typing import Union, Tuple

import rasterio
import geopandas as gpd


def create_basin_dem(basin_shp: str,
                     dem_file: str,
                     run_name: str,
                     write_raster: bool = False,
                     output_directory: Union[str, None] = None) -> Tuple:
    """Mask the input DEM using a basin geometry representative of the contributing area.

    :param basin_shp:               Full path with file name and extension to the target basin shapefile
    :type basin_shp:                str

    :param dem_file:                Full path with file name and extension to the input DEM raster file.
    :type dem_file:                 str

    :param run_name:                Name of run, all lowercase and only underscore separated.
    :type run_name:                 str

    :param write_raster:            Choice to write masked raster to file.
    :type write_raster:             bool

    :param output_directory:        Full path to a write-enabled directory to write output files to.  If write_raster
                                    is True, a valid path must be set.
    :type output_directory:         Union[str, None]

    :return:                        [0] an array of the 2D raster
                                    [1] The metadata dictionary of the raster

    """

    # dissolve target basin geometries
    basin_geom = gpd.read_file(basin_shp).dissolve().geometry.values[0]

    with rasterio.open(dem_file) as src:
        if src.crs is None:
            logging.warning("Input DEM raster does not have a defined coordinate reference system.")

        # apply basin geometry as a mask
        out_image, out_transform = rasterio.mask.mask(src, basin_geom, crop=True)

        # update the raster metadata with newly cropped extent
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # if writing raster to file
        if write_raster:

            if output_directory is None:
                msg = 'Please pass a value for output_directory if choosing to write shapefile outputs.'
                raise AssertionError(msg)

            output_file = os.path.join(output_directory, f"dem_masked_{run_name}.tif")
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)

            with rasterio.open(output_file) as get:
                arr = get.read(1)
                meta = get.meta

        else:

            # generate a masked raster file
            with tempfile.TemporaryDirectory() as tempdir:

                output_file = os.path.join(tempdir, f"dem_masked_{run_name}.tif")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(out_image)

                with rasterio.open(output_file) as get:
                    arr = get.read(1)
                    meta = get.meta

        return arr, meta
