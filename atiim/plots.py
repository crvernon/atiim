import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from typing import Tuple
from typing import Union
from scipy import stats

from .gage import import_gage_data


class PlotGeneral:
    """General plotting setup and option handling.

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    """

    def __init__(self,
                 style: str = 'whitegrid',
                 font_scale: float = 1.2,
                 figsize: Tuple[int] = (12, 8),
                 show_plot: bool = True,
                 save_plot: bool = False,
                 output_file: Union[str, None] = None,
                 dpi: int = 150):

        # set up plot style
        sns.set(style=style, font_scale=font_scale)

        # setup figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.output_file = output_file
        self.dpi = dpi

    def output_handler(self):
        """Handle plot display and output options."""

        # save figure
        if self.save_plot:

            # ensure a value is set for output_file
            if self.output_file is None:
                raise AssertionError("If writing plot to file, you must set a value for 'output_file'")

            plt.savefig(self.output_file, dpi=self.dpi)

        # show plot
        if self.show_plot:
            plt.show()


class PlotGageData(PlotGeneral):
    """Prepare gage data for plotting.

    :param gage_data:               Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type gage_data:                Union[str, pd.Dataframe]

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

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    """

    def __init__(self,
                 gage_data: Union[str, pd.DataFrame],
                 date_field_name: str = 'DATE',
                 time_field_name: str = 'TIME',
                 elevation_field_name: str = 'WL_ELEV_M',
                 style: str = 'whitegrid',
                 font_scale: float = 1.2,
                 figsize: Tuple[int] = (12, 8),
                 show_plot: bool = True,
                 save_plot: bool = False,
                 output_file: Union[str, None] = None,
                 dpi: int = 150):

        # initialize parent class
        super().__init__(style=style,
                         font_scale=font_scale,
                         figsize=figsize,
                         show_plot=show_plot,
                         save_plot=save_plot,
                         output_file=output_file,
                         dpi=dpi)

        self.gage_data = gage_data
        self.date_field_name = date_field_name
        self.time_field_name = time_field_name
        self.elevation_field_name = elevation_field_name

        # setup data frame
        self.df = self.prepare_data()

        self.z_max = self.df[elevation_field_name].max()
        self.z_min = self.df[elevation_field_name].min()
        self.n_records = self.df.shape[0]

        # sort elevation values in ascending order
        self.z_sorted = self.sorted_elevation()

        # calculate lognormal fit
        self.shape, self.location, self.scale = self.calculate_lognorm_fit()

    def prepare_data(self) -> pd.DataFrame:
        """Assign gage data to a data frame."""

        if type(self.gage_data) == pd.DataFrame:
            return self.gage_data

        else:
            return import_gage_data(self.gage_data, self.date_field_name, self.time_field_name)

    def generate_wse_interval(self, x_padding: float = 1.05, n_samples: int = 100) -> np.ndarray:
        """Generate an evenly spaced interval of values encompassing water surface elevation
        for use in scaling the x-axis.

        :param x_padding:               Multiplier for maximum elevation to determine an ending interval for the x-axis.
                                        E.g., if max value is 100 and x_padding is 1.1 then the ending bound would be 110.
        :type x_padding:                float

        :param n_samples:               The number of samples to generate over the x-axis space
        :type n_samples:                int

        :return:                        NumPy array of evenly spaced water surface elevation intervals

        """

        return np.linspace(self.df[self.elevation_field_name].min(),
                           self.df[self.elevation_field_name].max() * x_padding,
                           num=n_samples)

    def sorted_elevation(self) -> pd.Series:
        """Sort elevation data in ascending order."""

        return self.df[self.elevation_field_name].sort_values()

    def calculate_lognorm_fit(self) -> Tuple[float]:
        """Calculate the lognormal continuous random variable and generate parameter estimates. Uses
        Maximum Likelihood Estimateion (MLE).

        :returns:                               [0] shape parameter estimates
                                                [1] location parameters
                                                [2] scale parameters
        """

        return stats.lognorm.fit(self.z_sorted)

    def calculate_cdf(self, data: np.ndarray) -> np.ndarray:
        """Calculate the cumulative distribution.

        :param data:                    Input data for x-axis values
        :type data:                     np.ndarray

        :return:                        NumPy array of CDF outputs

        """

        return stats.lognorm.cdf(data, self.shape, self.location, self.scale)

    def calculate_pdf(self, data: np.ndarray):
        """Calculate the probability density.

        :param data:                    Input data for x-axis values
        :type data:                     np.ndarray

        :return:                        NumPy array of PDF outputs

        """

        return stats.lognorm.pdf(data, self.shape, self.location, self.scale)


def plot_wse_timeseries(gage_data: Union[str, pd.DataFrame],
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
                        color: str = '#005AB5',
                        transparency: float = 0.7):
    """Create plot for water surface elevation for the gage measurement period.

    :param gage_data:               Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type gage_data:                Union[str, pd.Dataframe]

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

    # prepare data for plotting
    data = PlotGageData(gage_data=gage_data,
                        date_field_name=date_field_name,
                        time_field_name=time_field_name,
                        elevation_field_name=elevation_field_name,
                        style=style,
                        font_scale=font_scale,
                        figsize=figsize,
                        show_plot=show_plot,
                        save_plot=save_plot,
                        output_file=output_file,
                        dpi=dpi)

    # construct a line plot
    sns.lineplot(x="date_time",
                 y=elevation_field_name,
                 data=data.df,
                 color=color,
                 alpha=transparency,
                 ax=data.ax)

    data.ax.set(ylabel='Water Surface Elevation (m)',
                xlabel=None,
                title='Water Surface Elevation Gage Measurements')

    # set limits for the x-axis
    plt.xlim(xmin=data.df['date_time'].min(), xmax=data.df['date_time'].max())

    plt.xticks(rotation=45)

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_cumulative_distribution(gage_data: Union[str, pd.DataFrame],
                                     show_plot: bool = True,
                                     save_plot: bool = False,
                                     output_file: Union[str, None] = None,
                                     dpi: int = 150,
                                     date_field_name: str = 'DATE',
                                     time_field_name: str = 'TIME',
                                     elevation_field_name: str = 'WL_ELEV_M',
                                     x_padding: float = 1.05,
                                     n_samples: int = 100,
                                     style: str = 'whitegrid',
                                     font_scale: float = 1.2,
                                     figsize: Tuple[int] = (12, 8),
                                     data_color: str = '#005AB5',
                                     lognorm_color: str = '#DC3220'):
    """Plot the cumulative distribution function for water surface elevation from the gage data.

    :param gage_data:               Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type gage_data:                Union[str, pd.Dataframe]

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param date_field_name:         Name of date field in file
    :type date_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

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

    # prepare data for plotting
    data = PlotGageData(gage_data=gage_data,
                        date_field_name=date_field_name,
                        time_field_name=time_field_name,
                        elevation_field_name=elevation_field_name,
                        style=style,
                        font_scale=font_scale,
                        figsize=figsize,
                        show_plot=show_plot,
                        save_plot=save_plot,
                        output_file=output_file,
                        dpi=dpi)

    # sort elevation value in ascending order
    z_sorted = data.sorted_elevation()

    # generate an evenly spaced interval of values encompassing water surface elevation for use in scaling the x-axis
    x_data = data.generate_wse_interval(x_padding=x_padding, n_samples=n_samples)

    # generate axis bounds and intervals for the cumulative distribution y-axis
    y_interval_array = np.linspace(0.0, 1.0, data.n_records)

    # plot elevation data series steps
    pd.Series(y_interval_array, index=z_sorted).plot(ax=data.ax, drawstyle='steps', label='data', color=data_color)

    # plot lognormal curve
    data.ax.plot(x_data, data.calculate_cdf(data=x_data), label='lognormal', color=lognorm_color)

    data.ax.set_xlabel('Water Surface Elevation (m)')
    data.ax.set_ylabel('CDF')
    data.ax.legend(loc=0, framealpha=0.5)
    plt.title('Cumulative Distribution of Water Surface Elevation')

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=data.z_max)

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_probability_density(gage_data: Union[str, pd.DataFrame],
                                 show_plot: bool = True,
                                 save_plot: bool = False,
                                 output_file: Union[str, None] = None,
                                 dpi: int = 150,
                                 date_field_name: str = 'DATE',
                                 time_field_name: str = 'TIME',
                                 elevation_field_name: str = 'WL_ELEV_M',
                                 x_padding: float = 1.05,
                                 n_samples: int = 100,
                                 style: str = 'whitegrid',
                                 font_scale: float = 1.2,
                                 figsize: Tuple[int] = (12, 8),
                                 data_color: str = '#005AB5',
                                 lognorm_color: str = '#DC3220',
                                 transparency: float = 0.6,
                                 n_bins: int = 21):
    """Plot the probability of occurrence for individual WSEs based on period-of-record.

    :param gage_data:               Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type gage_data:                Union[str, pd.Dataframe]

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param date_field_name:         Name of date field in file
    :type date_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

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

    :param transparency             Alpha value from 0 to 1 for transparency of histogram
    :type transparency              float

    :param n_bins:                  The number of equal-width bins in the range.
    :type n_bins:                   int

    """

    # prepare data for plotting
    data = PlotGageData(gage_data=gage_data,
                        date_field_name=date_field_name,
                        time_field_name=time_field_name,
                        elevation_field_name=elevation_field_name,
                        style=style,
                        font_scale=font_scale,
                        figsize=figsize,
                        show_plot=show_plot,
                        save_plot=save_plot,
                        output_file=output_file,
                        dpi=dpi)

    # sort elevation value in ascending order
    z_sorted = data.sorted_elevation()

    # generate an evenly spaced interval of values encompassing water surface elevation for use in scaling the x-axis
    x_data = data.generate_wse_interval(x_padding=x_padding, n_samples=n_samples)

    # plot binned elevation data
    data.ax.hist(z_sorted, bins=n_bins, label='data', density=True, color=data_color, alpha=transparency)

    # plot PDF outputs
    pdf = data.calculate_pdf(data=x_data)
    data.ax.plot(x_data, pdf, label='lognormal', color=lognorm_color)

    data.ax.set_xlabel('Water Surface Elevation (m)')
    data.ax.set_ylabel('Probability Density')
    data.ax.legend(loc=0, framealpha=0.5)
    plt.title('Probability Density of Water Surface Elevation')

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=x_data.max())

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_exceedance_probability(gage_data: Union[str, pd.DataFrame],
                 show_plot: bool = True,
                 save_plot: bool = False,
                 output_file: Union[str, None] = None,
                 dpi: int = 150,
                 date_field_name: str = 'DATE',
                 time_field_name: str = 'TIME',
                 elevation_field_name: str = 'WL_ELEV_M',
                 x_padding: float = 1.05,
                 n_samples: int = 100,
                 style: str = 'whitegrid',
                 font_scale: float = 1.2,
                 figsize: Tuple[int] = (12, 8),
                 data_color: str = '#005AB5',
                 lognorm_color: str = '#DC3220'):
    """Plot the probability of occurrence for individual WSEs based on period-of-record.


    """

    # prepare data for plotting
    data = PlotGageData(gage_data=gage_data,
                        date_field_name=date_field_name,
                        time_field_name=time_field_name,
                        elevation_field_name=elevation_field_name,
                        style=style,
                        font_scale=font_scale,
                        figsize=figsize,
                        show_plot=show_plot,
                        save_plot=save_plot,
                        output_file=output_file,
                        dpi=dpi)

    # sort elevation value in ascending order
    z_sorted = data.sorted_elevation()

    # generate an evenly spaced interval of values encompassing water surface elevation for use in scaling the x-axis
    x_data = data.generate_wse_interval(x_padding=x_padding, n_samples=n_samples)

    # calculate the lognormal continuous random variable and generate parameter estimates
    shape, location, scale = stats.lognorm.fit(z_sorted)

    # generate axis bounds and intervals for the cumulative dist y-axis
    cum_dist = np.linspace(0.0, 1.0, data.n_records)

    # plot elevation data series steps
    pd.Series(cum_dist, index=z_sorted).plot(ax=data.ax, drawstyle='steps', label='data', color=data_color)

    # plot lognormal curve
    data.ax.plot(x_data, stats.lognorm.cdf(x_data, shape, location, scale), label='lognormal', color=lognorm_color)

    data.ax.set_xlabel('Water Surface Elevation (m)')
    data.ax.set_ylabel('CDF')
    data.ax.legend(loc=0, framealpha=0.5)
    plt.title('Cumulative Distribution of Water Surface Elevation')

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=data.z_max)

    # handle display or output options
    data.output_handler()

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
                                  fill_color: str = '#005AB5',
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
