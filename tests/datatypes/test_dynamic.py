import datetime
import unittest

import numpy as np

from flightdataaccessor.datatypes.dynamic import (
    DateParameter, STORAGE_DATE_OFFSET, TimeParameter, get_date_array, get_datetime_array, get_time_array,
)


class TestFunctions(unittest.TestCase):
    def test_get_datetime_array(self):
        now = datetime.datetime.now()
        dta = get_datetime_array(now, 100)

        self.assertEqual(len(dta), 100)
        self.assertEqual(dta[0], np.datetime64(now, 's'))
        self.assertEqual(dta[-1], np.datetime64(now + datetime.timedelta(seconds=99), 's'))

    def test_get_date_array(self):
        today = datetime.datetime.now().date()
        # pick the beginning of the day to avoid roll-over
        today = datetime.datetime(today.year, today.month, today.day)
        dta = get_date_array(today, 100)

        self.assertEqual(len(dta), 100)
        self.assertEqual(dta[0] + STORAGE_DATE_OFFSET, np.datetime64(today.date()))
        self.assertEqual(dta[-1] + STORAGE_DATE_OFFSET, np.datetime64(today.date()))

    def test_get_time_array(self):
        now = datetime.datetime.now()
        today = datetime.datetime.now().date()
        today = datetime.datetime(today.year, today.month, today.day)
        time_offset = now - today
        dta = get_time_array(now, 100)

        self.assertEqual(len(dta), 100)
        # time array is an offset in seconds since midnight
        self.assertEqual(dta[0], np.timedelta64(time_offset, 's'))
        self.assertEqual(dta[-1], np.timedelta64(time_offset + datetime.timedelta(seconds=99), 's'))


class TestDateParameter(unittest.TestCase):
    def test_create(self):
        today = datetime.datetime.now().date()
        today = datetime.datetime(today.year, today.month, today.day)
        parameter = DateParameter(start_datetime=today, duration=100)

        self.assertEqual(parameter.name, 'Date')
        self.assertEqual(len(parameter.array), 100)
        parameter = DateParameter('Custom Date', start_datetime=today, duration=100)
        self.assertEqual(parameter.name, 'Custom Date')


class TestTimeParameter(unittest.TestCase):
    def test_create(self):
        now = datetime.datetime.now()
        parameter = TimeParameter(start_datetime=now, duration=100)
        self.assertEqual(parameter.name, 'Time')
        self.assertEqual(len(parameter.array), 100)
        parameter = TimeParameter('Custom Time', start_datetime=now, duration=100)
        self.assertEqual(parameter.name, 'Custom Time')
