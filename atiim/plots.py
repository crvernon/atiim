import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Tuple
from typing import Union
from scipy import stats


def plot_gage_wse(gage_data_file: str,
                  show_plot: bool = True,
                  save_plot: bool = False,
                  output_file: Union[str, None] = None,
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

    :param gage_data_file:          Full path with file name and extension to the gage data file
    :type gage_data_file:           str

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

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

    sns.lineplot(x="date_time",
                 y=elevation_field_name,
                 data=df,
                 color=color,
                 alpha=transparency,
                 ax=ax)

    ax.set(ylabel='Water Surface Elevation (m)',
           xlabel=None,
           title='Water Surface Elevation Gage Measurements')

    plt.xlim(xmin=df['date_time'].min(), xmax=df['date_time'].max())

    plt.xticks(rotation=45)

    # save figure
    if save_plot:

        # ensure a value is set for output_file
        if output_file is None:
            raise AssertionError("If writing plot to file, you must set a value for 'output_file'")

        plt.savefig(output_file, dpi=dpi)

    # show plot
    if show_plot:
        plt.show()

    plt.close()


def plot_wse_cdf(gage_data_file: str,
                 show_plot: bool = True,
                 save_plot: bool = False,
                 output_file: Union[str, None] = None,
                 dpi: int = 150,
                 elevation_field_name: str = 'WL_ELEV_M',
                 x_padding: float = 1.05,
                 n_samples: int = 100,
                 style: str = 'whitegrid',
                 font_scale: float = 1.2,
                 figsize: Tuple[int] = (12, 8),
                 data_color: str = 'blue',
                 lognorm_color: str = 'green'):
    """Plot the cumulative distribution function for water surface elevation from the gage data.

    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param elevation_field_name:    Name of elevation field in file
    :type elevation_field_name:     str

    :param x_padding:               Multiplier for maximum elevation to determine an ending interval for the x-axis.
                                    E.g., if max value is 100 and x_padding is 1.1 then the ending bound would be 110.
    :type x_padding:                float

    :param n_samples:               The number of samples to generate over the x-axis space
    :type n_samples:                int

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param data_color:              Color of data line
    :type data_color:               str

    :param lognorm_color:           Color of data line
    :type lognorm_color:            str

    """

    sns.set(style=style, font_scale=font_scale)

    fig, ax = plt.subplots(figsize=figsize)

    df = pd.read_csv(gage_data_file)

    # sorted elevation ascending
    z_sort = df[elevation_field_name].sort_values()

    # create x-axis steps for water elevation
    x_data = np.linspace(df[elevation_field_name].min(),
                         df[elevation_field_name].max() * x_padding,
                         num=n_samples)

    # calculate the lognormal continuous random variable and generate parameter estimates
    shape, location, scale = stats.lognorm.fit(z_sort)

    # generate axis bounds and intervals for the cumulative dist y-axis
    cum_dist = np.linspace(0.0, 1.0, df[elevation_field_name].shape[0])

    # plot elevation data series steps
    pd.Series(cum_dist, index=z_sort).plot(ax=ax, drawstyle='steps', label='data', color=data_color)

    # plot lognormal curve
    ax.plot(x_data, stats.lognorm.cdf(x_data, shape, location, scale), label='lognormal', color=lognorm_color)

    ax.set_xlabel('Water Surface Elevation (m)')
    ax.set_ylabel('CDF')
    ax.legend(loc=0, framealpha=0.5)
    plt.title('Cumulative Distribution of Water Surface Elevation')

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=df[elevation_field_name].max())

    # save figure
    if save_plot:

        # ensure a value is set for output_file
        if output_file is None:
            raise AssertionError("If writing plot to file, you must set a value for 'output_file'")

        plt.savefig(output_file, dpi=dpi)

    # show plot
    if show_plot:
        plt.show()

    plt.close()


def plot_hectare_hours_inundation(df: pd.DataFrame,
                                  show_plot: bool = True,
                                  save_plot: bool = False,
                                  output_file: Union[str, None] = None,
                                  dpi: int = 150,
                                  y_pad_fraction: float = 0.15,
                                  x_pad_fraction: float = 0.05,
                                  style: str = 'whitegrid',
                                  font_scale: float = 1.2,
                                  figsize: Tuple[int] = (12, 8),
                                  fill_color: str = 'blue',
                                  transparency: float = 0.7):
    """Plot of the hectare hours of inundation over water surface elevations.

    :param df:                      An data frame containing inundation data as a result of atiim.simulate_inundation()
    :type df:                       pd.DataFrame

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param y_pad_fraction:          A decimal fraction of the maximum elevation value to use as a padding on the Y axis
    :type y_pad_fraction:           float

    :param x_pad_fraction:          A decimal fraction of the maximum hectare hour value to use as a padding on the
                                    X axis
    :type x_pad_fraction:           float

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param fill_color:              Color of filled area in plot
    :type fill_color:               str

    :param transparency             Alpha value from 0 to 1 for transparency
    :type transparency              float

    """

    sns.set(style=style, font_scale=font_scale)

    fig, ax = plt.subplots(figsize=figsize)

    # pad min and max Y values for axis
    y_padding = df['elevation'].max() * y_pad_fraction

    # pad max x axis value
    x_padding = df['hect_hours'].max() * x_pad_fraction

    plt.ylim(ymin=(df['elevation'].min() - y_padding),
             ymax=(df['elevation'].max() + y_padding))

    plt.xlim(xmin=df['hect_hours'].min(),
             xmax=df['hect_hours'].max() + x_padding)

    plt.plot(df['hect_hours'], df['elevation'], 'black')

    plt.fill_betweenx(df['elevation'], df['hect_hours'], color=fill_color, alpha=transparency)

    plt.title('Hectare Hours of Inundation')
    plt.xlabel('Hectare Hours')
    plt.ylabel('Water Surface Elevation (m)')

    # save figure
    if save_plot:

        # ensure a value is set for output_file
        if output_file is None:
            raise AssertionError("If writing plot to file, you must set a value for 'output_file'")

        plt.savefig(output_file, dpi=dpi)

    # show plot
    if show_plot:
        plt.show()

    plt.close()


def plot_hypsometric(df: pd.DataFrame,
                     show_plot: bool = True,
                     save_plot: bool = False,
                     output_file: Union[str, None] = None,
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

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

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

    sns.lineplot(x=x_field_name,
                 y=y_field_name,
                 marker='o',
                 data=df,
                 color=color)

    ax.set(ylabel=y_label,
           xlabel=x_label,
           title=title)

    # format x axis label to bin by 1000
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:}'.format(int(x / 1000)) + 'K'))

    # save figure
    if save_plot:

        # ensure a value is set for output_file
        if output_file is None:
            raise AssertionError("If writing plot to file, you must set a value for 'output_file'")

        plt.savefig(output_file, dpi=dpi)

    # show plot
    if show_plot:
        plt.show()

    plt.close()
