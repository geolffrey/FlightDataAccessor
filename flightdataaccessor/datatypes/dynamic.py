"""
Dynamic parameters.

The "paraemters" that are used by the analysis that are redundant and easily reporoducible from other information and
hence not stored to save space.
"""

import datetime

import numpy as np

from flightdatautilities import units as ut

from .parameter import Parameter

STORAGE_DATE_OFFSET = np.datetime64('1990-01-01', 'D')


def get_datetime_array(start_datetime=None, duration=0):
    """Return Numpy array of datetimes at 1Hz frequency."""
    dt0 = np.datetime64(start_datetime or datetime.datetime.now(datetime.timezone.utc))
    dt1 = dt0 + np.timedelta64(int(duration), 's')
    return np.arange(dt0, dt1, dtype='datetime64[s]')


def get_date_array(start_datetime=None, duration=0):
    """Return Numpy array of dates at 1Hz frequency."""
    datetimes = get_datetime_array(start_datetime, duration)
    return datetimes.astype('datetime64[D]') - STORAGE_DATE_OFFSET


def get_time_array(start_datetime=None, duration=0):
    """Return Numpy array of time at 1Hz frequency."""
    datetimes = get_datetime_array(start_datetime, duration)
    dates = get_date_array(start_datetime, duration)
    return datetimes - dates - STORAGE_DATE_OFFSET.astype('datetime64[s]')


class DateParameter(Parameter):
    def __init__(self, name='Date', start_datetime=None, duration=0):
        array = get_date_array(start_datetime, duration)
        super(DateParameter, self).__init__(name, array=array, unit=ut.DAY, data_type='Unsigned')


class TimeParameter(Parameter):
    def __init__(self, name='Time', start_datetime=None, duration=0):
        array = get_time_array(start_datetime, duration)
        super(TimeParameter, self).__init__(name, array=array, unit=ut.SECOND, data_type='Unsigned')
