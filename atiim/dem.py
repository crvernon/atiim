import os
import logging
from typing import Union

import rasterio
import geopandas as gpd


def create_basin_dem(basin_shp: str,
                     dem_file: str,
                     run_name: str,
                     output_directory: Union[str, None] = None) -> str:
    """Mask the input DEM using a basin geometry representative of the contributing area.

    :param basin_shp:               Full path with file name and extension to the target basin shapefile
    :type basin_shp:                str

    :param dem_file:                Full path with file name and extension to the input DEM raster file.
    :type dem_file:                 str

    :param run_name:                Name of run, all lowercase and only underscore separated.
    :type run_name:                 str

    :param output_directory:        Full path to a write-enabled directory to write output files to
    :type output_directory:         Union[str, None]

    :return:                        Full path with file name and extension to the masked DEM raster file

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

        # write outputs
        output_file = os.path.join(output_directory, f"dem_masked_{run_name}.tif")
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

        return output_file
