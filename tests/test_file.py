from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from unittest.mock import patch

from flightdataaccessor.file import hdf_file


class TestHdfFile(unittest.TestCase):

    @patch('flightdataaccessor.formats.compatibility.open')
    def test_open(self, fdf_open):
        with self.assertWarns(DeprecationWarning):
            with hdf_file('test_fn'):
                fdf_open.assert_called_with('test_fn', mode='r+')
        with self.assertWarns(DeprecationWarning):
            with hdf_file('test_fn', read_only=True):
                fdf_open.assert_called_with('test_fn', mode='r')
        with self.assertWarns(DeprecationWarning):
            with hdf_file('test_fn', create=True):
                fdf_open.assert_called_with('test_fn', mode='x')

    @patch('flightdataaccessor.formats.compatibility.open')
    def test_open_fail(self, fdf_open):
        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                hdf_file('test_fn', create=True, read_only=True)
