import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Tuple
from typing import Union


def plot_hypsometric(df: pd.DataFrame,
                     output_file: str,
                     dpi: int = 150,
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

    :param df:                      A pandas data frame containing data to construct a hypsometric curve.
                                    See attim.hypsometric_curve()
    :type df:                       pd.DataFrame

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param x_field_name:            Field name of data for the x-axis
    :type x_field_name:             str

    :param y_field_name:            Field name of data for the y-axis
    :type y_field_name:             str

    :param x_label:                 Label for the x-axis
    :type x_label:                  str

    :param y_label:                 Label for the y-axis
    :type y_label:                  str

    :param title:                   Title for the plot if desired.  Use None for no title.
    :type title:                    str

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param color:                   Color of line and markers in plot
    :type color:                    str

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

    # save figure
    plt.savefig(output_file, dpi=dpi)

    plt.close()


def plot_gage_wse(gage_data_file: str,
                  output_file: str,
                  dpi: int = 150,
                  date_field_name: str = 'DATE',
                  time_field_name: str = 'TIME',
                  elevation_field_name: str = 'WL_ELEV_M',
                  style: str = 'whitegrid',
                  font_scale: float = 1.2,
                  figsize: Tuple[int] = (12, 8),
                  color: str = 'blue',
                  transparency: float = 0.7):
    """Create plot for water surface elevation for the gage measurement period.

    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param date_field_name:         Name of date field in file
    :type date_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

    :param elevation_field_name:    Name of elevation field in file
    :type elevation_field_name:     str

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param color:                   Color of line and markers in plot
    :type color:                    str

    :param transparency             Alpha value from 0 to 1 for transparency
    :type transparency              float

    """

    df = pd.read_csv(gage_data_file)

    # convert date and time strings to a pandas datetime type
    df['date_time'] = pd.to_datetime(df[date_field_name] + ' ' + df[time_field_name], infer_datetime_format=True)

    sns.set(style=style, font_scale=font_scale)

    fig, ax = plt.subplots(figsize=figsize)

    g = sns.lineplot(x="date_time",
                     y=elevation_field_name,
                     data=df,
                     color=color,
                     alpha=transparency)

    x = ax.set(ylabel='Water Surface Elevation (m)',
               xlabel=None,
               title='Water Surface Elevation Gage Measurements')

    plt.xlim(xmin=df['date_time'].min(), xmax=df['date_time'].max())

    plt.xticks(rotation=45)

    # save figure
    plt.savefig(output_file, dpi=dpi)

    plt.close()
