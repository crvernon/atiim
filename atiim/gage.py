import logging

import pandas as pd


def import_gage_data(gage_data_file: str,
                      date_field_name: str = 'DATE',
                      time_field_name: str = 'TIME'):
    """Process gage data tabular file.

    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param date_field_name:         Name of date field in file
    :type date_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

    :return:                        DataFrame of time series gage data sorted by date_time

    """

    # convert date and time strings to a pandas datetime type
    df = pd.read_csv(gage_data_file,  infer_datetime_format=True, parse_dates=[[date_field_name, time_field_name]])

    # rename date time field
    df.rename(columns={f'{date_field_name}_{time_field_name}': 'date_time'}, inplace=True)

    return df.sort_values(by=['date_time'])


def process_gage_data(gage_data_file: str,
                      date_field_name: str = 'DATE',
                      time_field_name: str = 'TIME',
                      elevation_field_name: str = 'WL_ELEV_M',
                      verbose: bool = False):
    """Process gage data tabular file.

    :param gage_data_file:          Full path with file name and extension to the gage data file.
    :type gage_data_file:           str

    :param date_field_name:         Name of date field in file
    :type date_field_name:          str

    :param time_field_name:         Name of time field in file
    :type time_field_name:          str

    :param elevation_field_name:    Name of elevation field in file
    :type elevation_field_name:     str

    :param verbose:                 Choice to log verbose description of file
    :type verbose:                  bool

    :returns:                       [0] minumum water elevation in file
                                    [1] maximum water elevation in file
                                    [2] dictionary of water elevation frequency {elev:  frequency}
    """

    # import gage data as time series
    df = import_gage_data(gage_data_file=gage_data_file,
                          date_field_name=date_field_name,
                          time_field_name=time_field_name)

    min_wtr_elev = df[elevation_field_name].min()
    max_wtr_elev = df[elevation_field_name].max()
    d_freq = df[elevation_field_name].value_counts().to_dict()

    if verbose:

        # calculate the number of days in the file
        n_days = (df['date_time'].max() - df['date_time'].min()).days

        logging.info(f"Total Time Steps:  {df.shape[0]}")
        logging.info(f"Days Verification:  {n_days}")
        logging.info(f"Minimum Water Surface Elevation:  {min_wtr_elev}")
        logging.info(f"Maximum Water Surface Elevation:  {max_wtr_elev}")

    return min_wtr_elev, max_wtr_elev, d_freq
