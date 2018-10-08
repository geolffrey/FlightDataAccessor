# TODO:
#  0. deidentify test data
#  1. test modify v2 file to use v2 attributes and 'series'
#  2. test modify v3 file to use v3 attributes and top level parameters

import os
import shutil
import unittest
import tempfile

import h5py
import numpy as np

from flightdataaccessor.formats.hdf import FlightDataFile
from flightdataaccessor.datatypes.parameter import Parameter


class FlightDataFileTestV2(unittest.TestCase):
    test_fn = 'data/flight_data_v2.hdf5'

    def get_data_from_hdf(self, hdf):
        '''Return the object storing parameters (depending on version)'''
        return hdf['series']

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        curr_dir = os.path.dirname(__file__)
        self.fp = os.path.join(self.tempdir, os.path.basename(self.test_fn))
        shutil.copy(os.path.join(curr_dir, self.test_fn), self.fp)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def getitem_test(self):
        with FlightDataFile(self.fp) as fdf:
            airs = fdf['Heading']
            self.assertEquals(airs.name, 'Heading')

    def setitem_test(self):
        """__setitem__ is equivalent to set_parameter()"""
        array = np.ma.arange(1000)
        param = Parameter('Test', array=array)
        with FlightDataFile(self.fp) as fdf:
            fdf['Test'] = param

        with h5py.File(self.fp, mode='r') as hdf:
            data = self.get_data_from_hdf(hdf)['Test']['data']
            self.assertTrue(np.all(param.array==data))

    def delitem_test(self):
        """__delitem__ is equivalent to delete_parameter()"""
        with FlightDataFile(self.fp) as fdf:
            self.assertIn('Airspeed', fdf)
            del fdf['Airspeed']

        with FlightDataFile(self.fp) as fdf:
            self.assertNotIn('Airspeed', fdf)

    def getattr_test(self):
        """Nonexistent attributes will return None"""
        with FlightDataFile(self.fp) as fdf:
            self.assertIsNone(fdf.nonexistent)
            self.assertTrue(fdf.reliable_frame_counter)

    def setattr_test(self):
        """Set attribute and read the value"""
        with FlightDataFile(self.fp) as fdf:
            fdf.new_attr = 123

        with FlightDataFile(self.fp) as fdf:
            self.assertEqual(fdf.new_attr, 123)

    def delattr_test(self):
        with FlightDataFile(self.fp) as fdf:
            del fdf.reliable_frame_counter

        with FlightDataFile(self.fp) as fdf:
            self.assertIsNone(fdf.reliable_frame_counter)

    def contains_test(self):
        with FlightDataFile(self.fp) as fdf:
            self.assertTrue('Airspeed' in fdf)

    def len_test(self):
        with FlightDataFile(self.fp) as fdf:
            self.assertTrue(len(fdf) == len(fdf.data.keys()))

    def keys_test(self):
        """
        Compare keys() with raw HDF5 keys() before and after adding a parameter
        """
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = self.get_data_from_hdf(hdf).keys()

        with FlightDataFile(self.fp) as fdf:
            old_keys = fdf.keys()
            self.assertEquals(hdf_series_names, old_keys)

            self.assertNotIn('Test', old_keys)
            param = Parameter('Test', array=np.ma.arange(1000))
            fdf.set_parameter(param)
            new_keys = fdf.keys()
            self.assertIn('Test', new_keys)

        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = self.get_data_from_hdf(hdf).keys()
            self.assertIn('Test', hdf_series_names)
            self.assertEquals(hdf_series_names, new_keys)

    def keys_valid_only_test(self):
        with FlightDataFile(self.fp) as fdf:
            valid_param_names = (p.name for p in fdf.values() if not p.invalid)
            self.assertItemsEqual(valid_param_names, fdf.keys(valid_only=True))

    def keys_subsets_test(self):
        with FlightDataFile(self.fp) as fdf:
    #         source_param_names = [p.name for p in fdf.values() if p.source]
    #         print len(source_param_names), len(fdf.keys(subset='source'))
    #         self.assertItemsEqual(source_param_names, fdf.keys(subset='source'))

            derived_param_names = [p.name for p in fdf.values() if not p.source]
            self.assertItemsEqual(derived_param_names, fdf.keys(subset='derived'))

            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source]
            self.assertItemsEqual(source_valid_param_names, fdf.keys(valid_only=True, subset='source'))

            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and not p.source]
            self.assertItemsEqual(derived_valid_param_names, fdf.keys(valid_only=True, subset='derived'))

    #         self.assertItemsEqual(derived_param_names + source_param_names, fdf.keys())
    #         self.assertItemsEqual(derived_valid_param_names + source_valid_param_names, fdf.keys(valid_only=True))

    def values_test(self):
        """
        Compare parameter names extracted with values() with raw HDF5 keys() (series names) before and after adding a
        parameter.
        """
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = self.get_data_from_hdf(hdf).keys()

        with FlightDataFile(self.fp) as fdf:
            fdf_param_names = (p.name for p in fdf.values())
            self.assertItemsEqual(hdf_series_names, fdf_param_names)

            param = Parameter('Test', array=np.ma.arange(1000))
            self.assertNotIn(param.name, fdf_param_names)
            fdf.set_parameter(param)
            fdf_param_names = (p.name for p in fdf.values())

        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = (self.get_data_from_hdf(hdf).keys())

        self.assertIn(param.name, hdf_series_names)

    def items_test(self):
        """
        Compare parameter names extracted with values() with raw HDF5 keys() (series names) before and after adding a
        parameter.
        """
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = set(self.get_data_from_hdf(hdf).keys())

        with FlightDataFile(self.fp) as fdf:
            param_names = set()
            for param_name, param in fdf.items():
                self.assertEquals(param_name, param.name)
                self.assertIn(param_name, hdf_series_names)
                param_names.add(param_name)

        self.assertEquals(param_names, hdf_series_names)

    def close_test(self):
        """
        Verify that HDF file is closed on FlightDataFile.close()
        """
        fdf = FlightDataFile(self.fp)
        fdf.keys()

        fdf.close()
        # h5py raises ValueError on access to a closed file
        with self.assertRaises(ValueError):
            fdf.hdf.keys()

    def get_parameter_test(self):
        array = np.ma.arange(1000)
        param = Parameter('Test', array=array)
        with FlightDataFile(self.fp) as fdf:
            fdf.set_parameter(param)

        with FlightDataFile(self.fp) as fdf:
            param = fdf.get_parameter('Test')
            self.assertTrue(np.all(param.array==array))

    def get_parameter_cache_test(self):
        """
        Get parameter, the retrieved object must be the same every time

        copy_param=False is required for comparison purposes.
        """
        with FlightDataFile(self.fp) as fdf:
            param1 = fdf.get_parameter('Airspeed', copy_param=False)
            self.assertIn(param1.name, fdf.parameter_cache)
            cached = fdf.parameter_cache['Airspeed']
            self.assertEquals(param1, cached)
            param2 = fdf.get_parameter('Airspeed', copy_param=False)
            self.assertEquals(param2, cached)

    def get_parameter_copy_test(self):
        """Get parameter with copy_param twice, the copies should differ"""

        with FlightDataFile(self.fp) as fdf:
            param1 = fdf.get_parameter('Airspeed', copy_param=True)
            param2 = fdf.get_parameter('Airspeed', copy_param=True)
            self.assertNotEquals(param1, param2)

    def set_parameter_test(self):
        """Set parameter, compare with raw data in HDF5"""

        array = np.ma.arange(1000)
        param = Parameter('Test', array=array)
        with FlightDataFile(self.fp) as fdf:
            fdf.set_parameter(param)

        with h5py.File(self.fp, mode='r') as hdf:
            data = self.get_data_from_hdf(hdf)['Test']['data']
            self.assertTrue(np.all(param.array==data))

    def delete_parameter_test(self):
        with FlightDataFile(self.fp) as fdf:
            self.assertIn('Airspeed', fdf)
            fdf.delete_parameter('Airspeed')

        with FlightDataFile(self.fp) as fdf:
            self.assertNotIn('Airspeed', fdf)

    def get_parameters_test(self):
        parameter_names = ('Airspeed', 'Heading')
        with FlightDataFile(self.fp) as fdf:
            parameters = fdf.get_parameters(parameter_names)

        self.assertItemsEqual(parameter_names, (p.name for p in parameters))

    def set_parameters_test(self):
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(1000)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFile(self.fp) as fdf:
            fdf.set_parameters(params)
            for name in parameter_names:
                self.assertIn(name, fdf)

    def delete_parameters_test(self):
        parameter_names = ('Airspeed', 'Heading')
        with FlightDataFile(self.fp) as fdf:
            fdf.delete_parameters(parameter_names)
            for name in parameter_names:
                self.assertNotIn(name, fdf)

    def parameter_cache_test(self):
        with FlightDataFile(self.fp) as fdf:
            self.assertFalse(fdf.parameter_cache)
            airs = fdf['Airspeed']
            self.assertIn('Airspeed', fdf.parameter_cache)


class FlightDataFileTestV3(FlightDataFileTestV2):
    test_fn = 'data/flight_data_v3.hdf5'

    def get_data_from_hdf(self, hdf):
        '''Return the object storing parameters (depending on version)'''
        return hdf
