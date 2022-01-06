from .gage import import_gage_data, process_gage_data
from .hypsometric import hypsometric_curve
from .inundation import simulate_inundation
from .plots import plot_wse_timeseries, plot_wse_cumulative_distribution, plot_wse_probability_density, \
    plot_wse_exceedance_probability, plot_inundation_hectare_hours, plot_inundation_perimeter, plot_inundation_area, \
    plot_hypsometric_curve

from .package_data import SampleData


__version__ = "0.0.0"
