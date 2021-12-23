import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Tuple


def hypsometric_curve(dem_file: str,
                      elevation_interval: float,
                      min_elevation: float = None,
                      max_elevation: float = None) -> pd.DataFrame:
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

    :param min_elevation:           Optional.  Maximum elevation to sample from.  Default is to use the maximum
                                    elevation of the raster as a starting point.
    :type min_elevation:            float

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


def hypsometric_plot(df: pd.DataFrame,
                     x_field_name: str = "dem_area_at_elevation",
                     y_field_name: str = "dem_elevation",
                     x_label: str = 'Area (m$^2$)',
                     y_label: str = 'Elevation (m)',
                     title: str = 'Hypsometric Curve',
                     style: str = 'whitegrid',
                     font_scale: float = 1.2,
                     figsize: Tuple[int] = (12, 8),
                     color: str = 'black'):
    """Plot a hypsometric curve as an elevation-area relationship assessment metric
    of the landform shape at a site.  Provides basic metric of opportunity for inundation and
    habitat opportunity.

    :param df:                  A pandas data frame containing data to construct a hypsometric curve.
                                See attim.hypsometric_curve()
    :type df:                   pd.DataFrame

    :param x_field_name:        Field name of data for the x-axis
    :type x_field_name:         str

    :param y_field_name:        Field name of data for the y-axis
    :type y_field_name:         str

    :param x_label:             Label for the x-axis
    :type x_label:              str

    :param y_label:             Label for the y-axis
    :type y_label:              str

    :param title:               Title for the plot if desired.  Use None for no title.
    :type title:                str

    :param style:               Seaborn style designation
    :type style:                str

    :param font_scale:          Scaling factor for font size
    :type font_scale:           float

    :param figsize:             Tuple of figure size (x, y)
    :type figsize:              Tuple[int]

    :param color:               Color of line and markers in plot
    :type color:                str
    
    """

    # seaborn settings
    sns.set(style=style, font_scale=font_scale)

    # setup figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    g = sns.lineplot(x=x_field_name,
                     y=y_field_name,
                     marker='o',
                     data=df,
                     color=color)

    x = ax.set(ylabel=y_label,
               xlabel=x_label,
               title=title)

    # format x axis label to bin by 1000
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:}'.format(int(x / 1000)) + 'K'))
