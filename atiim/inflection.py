from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def calculate_bankfull_elevation(df: pd.DataFrame,
                                 smooth_data: bool = False,
                                 smooth_sigma: int = 100,
                                 area_field_name: str = 'area',
                                 elevation_field_name: str = 'elevation') -> Tuple[float]:
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

    :returns:                       [0] Bankfull elevation value.  First inflection point.
                                    [1] Bankfull area value

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

    # bankfull area corresponding to the inflection point
    bankfull_area = df[area_field_name].values[inflection_indices[0]]

    return bankfull_elevation, bankfull_area
