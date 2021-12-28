from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FixedLocator

from .gage import import_gage_data
from .inflection import calculate_bankfull_elevation


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
                 data: Union[str, pd.DataFrame],
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

        self.data = data
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

        # set cumulative y-axis values
        self.y_cumulative_series = self.set_cumulative_series()

        # calculate lognormal fit
        self.shape, self.location, self.scale = self.calculate_lognorm_fit()

    def prepare_data(self) -> pd.DataFrame:
        """Assign gage data to a data frame."""

        if type(self.data) == pd.DataFrame:
            return self.data

        else:
            return import_gage_data(self.data, self.date_field_name, self.time_field_name)

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

    def calculate_pdf(self, data: np.ndarray) -> np.ndarray:
        """Calculate the probability density.

        :param data:                    Input data for x-axis values
        :type data:                     np.ndarray

        :return:                        NumPy array of PDF outputs

        """

        return stats.lognorm.pdf(data, self.shape, self.location, self.scale)

    def set_cumulative_series(self) -> pd.Series:
        """Generate axis series and intervals for the cumulative distribution y-axis."""

        # generate axis bounds and intervals for the cumulative distribution y-axis
        y_cumulative = np.linspace(0.0, 1.0, self.n_records)

        return pd.Series(y_cumulative, index=self.z_sorted)

    def calculate_exceedance_probability(self) -> pd.Series:
        """Calculate a y-axis series of exceedance probilites as a percentage."""

        return (1.0 - self.y_cumulative_series) * 100


class PlotInundationData(PlotGeneral):
    """Prepare inundation data for plotting.

    :param data:                    Either a full path with file name and extension to the inundation data file or
                                    a data frame processed using atiim.simulate_inundation()
    :type data:                     Union[str, pd.Dataframe]

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

    # field names in inundation standard output
    FLD_ID = 'id'
    FLD_FREQUENCY = 'frequency'
    FLD_ELEVATION = 'elevation'
    FLD_AREA = 'area'
    FLD_HECTARES = 'hectares'
    FLD_PERIMETER = 'perimeter'
    FLD_HECTARE_HOURS = 'hect_hours'

    def __init__(self,
                 data: Union[str, pd.DataFrame],
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

        self.data = data

        # read data in as data frame
        self.df = self.prepare_data()

        # unpack fields
        self.id = self.df[PlotInundationData.FLD_ID]
        self.frequency = self.df[PlotInundationData.FLD_FREQUENCY]
        self.elevation = self.df[PlotInundationData.FLD_ELEVATION]
        self.area = self.df[PlotInundationData.FLD_AREA]
        self.hectares = self.df[PlotInundationData.FLD_HECTARES]
        self.perimeter = self.df[PlotInundationData.FLD_PERIMETER]
        self.hectare_hours = self.df[PlotInundationData.FLD_HECTARE_HOURS]

    def prepare_data(self) -> pd.DataFrame:
        """Assign input data to a data frame."""

        if type(self.data) == pd.DataFrame:
            return self.data

        else:
            return pd.read_csv(self.data)

    def set_xaxis_to_thousands(self):
        """Format the x-axis to show thousands as K (e.g., 15,000 as 15K)."""

        # format the x-axis to show thousands as K (e.g., 15,000 as 15K)
        self.ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: '{:}'.format(int(x / 1000)) + 'K' if x >= 1000 else int(x))
        )


def plot_wse_timeseries(data: Union[str, pd.DataFrame],
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
                        transparency: float = 0.7,
                        title: Union[str, None] = 'Water Surface Elevation Gage Measurements'):
    """Create plot for water surface elevation for the gage measurement period.

    :param data:                    Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type data:                     Union[str, pd.Dataframe]

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

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotGageData(data=data,
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
                title=title)

    # set limits for the x-axis
    plt.xlim(xmin=data.df['date_time'].min(), xmax=data.df['date_time'].max())

    plt.xticks(rotation=45)

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_cumulative_distribution(data: Union[str, pd.DataFrame],
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
                                     title: Union[str, None] = 'Cumulative Distribution of Water Surface Elevation'):
    """Plot the cumulative distribution function for water surface elevation from the gage data.

    :param data:                     Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type data:                     Union[str, pd.Dataframe]

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

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotGageData(data=data,
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

    # generate an evenly spaced interval of values encompassing water surface elevation for use in scaling the x-axis
    x_data = data.generate_wse_interval(x_padding=x_padding, n_samples=n_samples)

    # plot elevation data series steps
    data.y_cumulative_series.plot(ax=data.ax, drawstyle='steps', label='data', color=data_color)

    # plot lognormal curve
    data.ax.plot(x_data, data.calculate_cdf(data=x_data), label='lognormal', color=lognorm_color)

    data.ax.set_xlabel('Water Surface Elevation (m)')
    data.ax.set_ylabel('Cumulative Distribution')
    data.ax.legend(loc=0, framealpha=0.5)
    plt.title(title)

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=data.z_max)

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_probability_density(data: Union[str, pd.DataFrame],
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
                                 n_bins: int = 21,
                                 title: Union[str, None] = 'Probability Density of Water Surface Elevation'):
    """Plot the probability density of water surface elevations based on the period-of-record.

    :param data:                    Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type data:                     Union[str, pd.Dataframe]

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

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotGageData(data=data,
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
    plt.title(title)

    # set x-axis limits
    plt.xlim(xmin=x_data.min(), xmax=x_data.max())

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_wse_exceedance_probability(data: Union[str, pd.DataFrame],
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
                                    title: Union[str, None] = 'Exceedance Probability of Water Surface Elevation'):
    """Plot the probability of occurrence for individual water surface elevations based on the period-of-record.

    :param data:                    Either a full path with file name and extension to the gage data file or
                                    a data frame processed using atiim.import_gage_data()
    :type data:                     Union[str, pd.Dataframe]

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

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotGageData(data=data,
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

    # plot exceedance probability as a percentage
    exceedance_probability_series = data.calculate_exceedance_probability()
    data.ax.semilogx(exceedance_probability_series, z_sorted, ls='', marker='o', label='data', color=data_color)

    # plot cumulative distribution outputs
    cumulative = 100 * (1.0 - data.calculate_cdf(data=x_data))
    data.ax.plot(cumulative, x_data, label='lognormal', color=lognorm_color)

    # set interval spacing and formatting of the x-axis
    minor_locator = FixedLocator([1, 2, 5, 10, 20, 50, 100])
    data.ax.xaxis.set_major_locator(minor_locator)
    data.ax.xaxis.set_major_formatter(ScalarFormatter())
    data.ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))

    # set axis limits
    data.ax.set_xlim(1, 100)
    data.ax.set_ylim(0, data.z_max)

    data.ax.set_xlabel('Exceedance Probability (%)')
    data.ax.set_ylabel('Water Surface Elevation (m)')
    data.ax.invert_xaxis()
    data.ax.legend(loc=0, framealpha=0.5)

    plt.title(title)

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_inundation_hectare_hours(data: Union[str, pd.DataFrame],
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
                                  transparency: float = 0.7,
                                  title: Union[str, None] = 'Hectare Hours of Inundation'):
    """Plot of the hectare hours of inundation over water surface elevations.

    :param data:                    An data frame or CSV file containing inundation data as a result of
                                    atiim.simulate_inundation()
    :type data:                     Union[str, pd.DataFrame]

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

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotInundationData(data=data,
                              style=style,
                              font_scale=font_scale,
                              figsize=figsize,
                              show_plot=show_plot,
                              save_plot=save_plot,
                              output_file=output_file,
                              dpi=dpi)

    # pad min and max Y values for axis
    z_max = data.elevation.max()
    z_min = data.elevation.min()
    y_padding = z_max * y_pad_fraction

    # pad max x axis value
    h_max = data.hectare_hours.max()
    h_min = data.hectare_hours.min()
    x_padding = h_max * x_pad_fraction

    # set axis limits
    plt.ylim(ymin=(z_min - y_padding),
             ymax=(z_max + y_padding))

    plt.xlim(xmin=h_min,
             xmax=h_max + x_padding)

    plt.plot(data.hectare_hours, data.elevation, 'black')

    # fill area under the curve
    plt.fill_betweenx(data.elevation, data.hectare_hours, color=fill_color, alpha=transparency)

    plt.title(title)
    plt.xlabel('Hectare Hours')
    plt.ylabel('Water Surface Elevation (m)')

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_inundation_perimeter(data: Union[str, pd.DataFrame],
                              show_plot: bool = True,
                              save_plot: bool = False,
                              output_file: Union[str, None] = None,
                              dpi: int = 150,
                              x_pad_fraction: float = 0.05,
                              style: str = 'whitegrid',
                              font_scale: float = 1.2,
                              figsize: Tuple[int] = (12, 8),
                              title: Union[str, None] = 'Inundation Perimeter by Water Surface Elevation'):
    """Plot perimeter of inundation through the water surface elevation time series.

    :param data:                    An data frame or CSV file containing inundation data as a result of
                                    atiim.simulate_inundation()
    :type data:                     Union[str, pd.DataFrame]

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param x_pad_fraction:          A decimal fraction of the maximum hectare hour value to use as a padding on the
                                    X axis
    :type x_pad_fraction:           float

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    """

    # prepare data for plotting
    data = PlotInundationData(data=data,
                              style=style,
                              font_scale=font_scale,
                              figsize=figsize,
                              show_plot=show_plot,
                              save_plot=save_plot,
                              output_file=output_file,
                              dpi=dpi)

    # plot perimeter line with points
    plt.plot(data.perimeter, data.elevation, 'ko', data.perimeter, data.elevation, 'k')

    # plot mean elevation line
    plt.axhline(data.elevation.mean(), color='black', linestyle="--", label='_nolegend_')

    p_max = data.perimeter.max()
    x_padding = p_max * x_pad_fraction
    plt.xlim(xmin=0, xmax=p_max + x_padding)

    data.ax.set(ylabel='Water Surface Elevation (m)',
                xlabel='Inundation Perimeter (m)',
                title=title)

    # format the x-axis to show thousands as K (e.g., 15,000 as 15K)
    data.set_xaxis_to_thousands()

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_inundation_area(data: Union[str, pd.DataFrame],
                         show_plot: bool = True,
                         save_plot: bool = False,
                         output_file: Union[str, None] = None,
                         dpi: int = 150,
                         x_pad_fraction: float = 0.05,
                         style: str = 'whitegrid',
                         font_scale: float = 1.2,
                         figsize: Tuple[int] = (12, 8),
                         title: Union[str, None] = 'Inundation Area by Water Surface Elevation',
                         inflection_pt_size: int = 14):
    """Plot area of inundation through the water surface elevation time series.

    :param data:                    An data frame or CSV file containing inundation data as a result of
                                    atiim.simulate_inundation()
    :type data:                     Union[str, pd.DataFrame]

    :param show_plot:               If True, plot will be displayed
    :type show_plot:                bool

    :param save_plot:               If True, plot will be written to file and a value must be set for output_file
    :type save_plot:                bool

    :param output_file:             Full path with file name and extension to an output file
    :type output_file:              str

    :param dpi:                     The resolution in dots per inch
    :type dpi:                      int

    :param x_pad_fraction:          A decimal fraction of the maximum hectare hour value to use as a padding on the
                                    X axis
    :type x_pad_fraction:           float

    :param style:                   Seaborn style designation
    :type style:                    str

    :param font_scale:              Scaling factor for font size
    :type font_scale:               float

    :param figsize:                 Tuple of figure size (x, y)
    :type figsize:                  Tuple[int]

    :param title                    Plot title.  If None, then no title will be displayed.
    :type title                     Union[str, None]

    :param inflection_pt_size:      Size of the inflection point marker
    :type inflection_pt_size:       int

    """

    # prepare data for plotting
    data = PlotInundationData(data=data,
                              style=style,
                              font_scale=font_scale,
                              figsize=figsize,
                              show_plot=show_plot,
                              save_plot=save_plot,
                              output_file=output_file,
                              dpi=dpi)

    # calculate the inflection point for bankfull elevation
    bankfull_elevation, bankfull_area = calculate_bankfull_elevation(df=data.df)

    # plot line with points for inundated area by water surface elevation
    plt.plot(data.area, data.elevation, 'ko', data.area, data.elevation, 'k')

    # plot the point for bankfull elevation
    plt.plot(bankfull_area, bankfull_elevation, 'r^', ms=inflection_pt_size)

    data.ax.set(ylabel='Water Surface Elevation (m)',
                xlabel='Area (m$^2$)',
                title=title)

    # format the x-axis to show thousands as K (e.g., 15,000 as 15K)
    data.set_xaxis_to_thousands()

    # handle display or output options
    data.output_handler()

    plt.close()


def plot_hypsometric_curve(df: pd.DataFrame,
                           show_plot: bool = True,
                           save_plot: bool = False,
                           output_file: Union[str, None] = None,
                           dpi: int = 150,
                           x_field_name: str = "dem_area_at_elevation",
                           y_field_name: str = "dem_elevation",
                           x_label: str = 'Area (m$^2$)',
                           y_label: str = 'Elevation (m)',
                           title: Union[str, None] = 'Hypsometric Curve',
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
    :type title:                    Union[str, None]

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
