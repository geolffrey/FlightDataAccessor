# TODO:
#  0. deidentify test data
#  1. test modify v2 file to use v2 attributes and 'series'
#  2. test modify v3 file to use v3 attributes and top level parameters

import json
import math
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from flightdataaccessor.datatypes.parameter import Parameter
from flightdataaccessor.formats.hdf import FlightDataFile


class FlightDataFileTestV2(unittest.TestCase):
    test_fn = 'data/flight_data_v2.hdf5'

    def assertItemsEqual(self, l1, l2):
        self.assertEqual(list(l1), list(l2))

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

    def test_file_mode_r(self):
        """Open file in 'r' mode."""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFile(self.fp) as fdf:
            # read-only (default), assignment fails
            with self.assertRaises(IOError):
                fdf['Test'] = param
            with self.assertRaises(IOError):
                fdf.reliable_frame_counter = not fdf.reliable_frame_counter

        with self.assertRaises(OSError):
            FlightDataFile(self.fp + '-nonexistent')

    def test_file_mode_w(self):
        """Open file in 'w' mode."""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFile(self.fp, mode='w') as fdf:
            # existing file will be wiped
            self.assertFalse(fdf)
            fdf['Test'] = param
            self.assertIn('Test', fdf)

        with FlightDataFile(self.fp) as fdf:
            # contains our changes
            self.assertIn('Test', fdf)

        with FlightDataFile(self.fp + '-nonexistent', mode='w') as fdf:
            # not existing file will be opened
            self.assertFalse(fdf)
            fdf['Test'] = param
            self.assertIn('Test', fdf)

        with FlightDataFile(self.fp + '-nonexistent') as fdf:
            # still exists and contains our changes
            self.assertIn('Test', fdf)

    def test_file_mode_a(self):
        """Open file in 'a' mode."""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFile(self.fp, 'a') as fdf:
            # append-or-create is the least restrictive mode
            fdf['Test'] = param
            rfc = fdf.reliable_frame_counter
            fdf.reliable_frame_counter = not fdf.reliable_frame_counter

            self.assertIn('Test', fdf)
            self.assertEqual(fdf.reliable_frame_counter, not rfc)

        with FlightDataFile(self.fp) as fdf:
            self.assertIn('Test', fdf)
            self.assertEqual(fdf.reliable_frame_counter, not rfc)

        with FlightDataFile(self.fp + '-nonexistent', 'a') as fdf:
            # the file is empty
            self.assertFalse(fdf)

    def test_file_mode_x(self):
        """Open file in 'x' mode."""
        with self.assertRaises(IOError):
            # existing file will not be opened
            FlightDataFile(self.fp, 'x')

        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFile(self.fp + '-nonexistent', mode='x') as fdf:
            # not existing file will get created
            self.assertFalse(fdf)
            fdf['Test'] = param

        with FlightDataFile(self.fp + '-nonexistent') as fdf:
            # still exists and contains our changes
            self.assertIn('Test', fdf)

    def test_getitem(self):
        """Ensure the name of retrieved parameter is correct."""
        with FlightDataFile(self.fp) as fdf:
            airs = fdf['Heading']
            self.assertEqual(airs.name, 'Heading')

    def test_setitem(self):
        """__setitem__ is equivalent to set_parameter()"""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf['Test'] = param

        with h5py.File(self.fp) as hdf:
            data = self.get_data_from_hdf(hdf)['Test']['data']
            self.assertTrue(np.all(param.array == data))

    def test_delitem(self):
        """__delitem__ is equivalent to delete_parameter()"""
        with FlightDataFile(self.fp, mode='a') as fdf:
            self.assertIn('Airspeed', fdf)
            del fdf['Airspeed']
            with self.assertRaises(KeyError):
                del fdf['Airspeed']
            self.assertNotIn('Airspeed', fdf)

        with FlightDataFile(self.fp) as fdf:
            self.assertNotIn('Airspeed', fdf)

    def test_getattr(self):
        """Nonexistent attributes will return None."""
        with FlightDataFile(self.fp) as fdf:
            self.assertTrue(fdf.reliable_frame_counter)
            with self.assertRaises(AttributeError):
                fdf.nonexistent

    def test_setattr(self):
        """Set attribute and read the value."""
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.new_attr = 123

        with FlightDataFile(self.fp) as fdf:
            self.assertEqual(fdf.new_attr, 123)

    def test_delattr(self):
        """Ensure removed attribute returns None."""
        with FlightDataFile(self.fp, mode='a') as fdf:
            del fdf.reliable_frame_counter

        with FlightDataFile(self.fp, 'a') as fdf:
            with self.assertRaises(AttributeError):
                del fdf.reliable_frame_counter

    def test_contains(self):
        """Ensure `in` operator works for the object."""
        with FlightDataFile(self.fp) as fdf:
            self.assertTrue('Airspeed' in fdf)

    def test_len(self):
        """Compare len of the FlightDataFile object with the len of data series in raw HDF file."""
        with FlightDataFile(self.fp) as fdf:
            self.assertTrue(len(fdf) == len(fdf.data.keys()))

    def test_keys(self):
        """Test the consistency of keys cache for all parameters."""
        with FlightDataFile(self.fp) as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(sorted(derived_param_names + lfl_param_names), sorted(fdf.keys()))

    def test_keys_valid_only(self):
        """Test the consistency of keys cache for valid parameters."""
        with FlightDataFile(self.fp) as fdf:
            valid_param_names = (p.name for p in fdf.values() if not p.invalid)
            self.assertItemsEqual(valid_param_names, fdf.keys(valid_only=True))

            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source]
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and not p.source]
            self.assertItemsEqual(derived_valid_param_names + source_valid_param_names, fdf.keys(valid_only=True))

    def test_keys_source(self):
        """Test the consistency of keys cache for source parameters."""
        with FlightDataFile(self.fp) as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            self.assertItemsEqual(lfl_param_names, fdf.keys(subset='lfl'))

    def test_keys_derived(self):
        """Test the consistency of keys cache for derived parameters."""
        with FlightDataFile(self.fp) as fdf:
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(derived_param_names, fdf.keys(subset='derived'))

    def test_keys_valid_source(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFile(self.fp) as fdf:
            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'lfl']
            self.assertItemsEqual(source_valid_param_names, fdf.keys(valid_only=True, subset='lfl'))

    def test_keys_valid_derived(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFile(self.fp) as fdf:
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'derived']
            self.assertItemsEqual(derived_valid_param_names, fdf.keys(valid_only=True, subset='derived'))

    def test_keys_add(self):
        """Compare keys() with raw HDF5 keys() before and after adding a parameter."""
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = list(self.get_data_from_hdf(hdf).keys())

        with FlightDataFile(self.fp, mode='a') as fdf:
            old_keys = fdf.keys()
            self.assertEqual(hdf_series_names, old_keys)

            self.assertNotIn('Test', old_keys)
            param = Parameter('Test', array=np.ma.arange(100))
            fdf.set_parameter(param)
            new_keys = fdf.keys()
            self.assertIn('Test', new_keys)

        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = self.get_data_from_hdf(hdf).keys()
            self.assertIn('Test', hdf_series_names)
            self.assertItemsEqual(hdf_series_names, new_keys)

    def test_values(self):
        """
        Compare parameter names extracted with values() with raw HDF5 keys() (series names) before and after adding a
        parameter.
        """
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = list(self.get_data_from_hdf(hdf).keys())

        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf_param_names = (p.name for p in fdf.values())
            self.assertItemsEqual(hdf_series_names, fdf_param_names)

            param = Parameter('Test', array=np.ma.arange(100))
            self.assertNotIn(param.name, fdf_param_names)
            fdf.set_parameter(param)
            fdf_param_names = (p.name for p in fdf.values())

        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = list(self.get_data_from_hdf(hdf).keys())

        self.assertIn(param.name, hdf_series_names)

    def test_items(self):
        """
        Compare parameter names extracted with values() with raw HDF5 keys() (series names) before and after adding a
        parameter.
        """
        with h5py.File(self.fp, mode='r') as hdf:
            hdf_series_names = set(self.get_data_from_hdf(hdf).keys())

        with FlightDataFile(self.fp) as fdf:
            param_names = set()
            for param_name, param in fdf.items():
                self.assertEqual(param_name, param.name)
                self.assertIn(param_name, hdf_series_names)
                param_names.add(param_name)

        self.assertEqual(param_names, hdf_series_names)

    def test_duration(self):
        """Verify that file duration is calculated automatically on close."""
        with FlightDataFile(self.fp) as fdf:
            airs = fdf['Airspeed']

        with FlightDataFile(self.fp + '-copy', 'x') as fdf:
            # copy first 100s of data
            fdf['Airspeed'] = airs.trim(0, 100)

        with FlightDataFile(self.fp + '-copy') as fdf:
            self.assertEqual(fdf.duration, 100)

    def test_frequencies(self):
        """Verify that file frequencies is calculated automatically on close."""
        with FlightDataFile(self.fp) as fdf:
            airs = fdf['Airspeed']
            accel = fdf['Acceleration Normal']

        with FlightDataFile(self.fp + '-copy', 'x') as fdf:
            # copy first 100s of data
            fdf['Airspeed'] = airs
            fdf['Acceleration Normal'] = accel

        with FlightDataFile(self.fp + '-copy') as fdf:
            np.testing.assert_array_equal(sorted(fdf.frequencies), sorted([airs.frequency, accel.frequency]))

    def test_close(self):
        """Verify that HDF file is closed on FlightDataFile.close()."""
        fdf = FlightDataFile(self.fp)
        fdf.keys()

        fdf.close()
        # after closing the file None is assigned to it
        with self.assertRaises(AttributeError):
            fdf.file.keys()

    def test_get_parameter(self):
        """Compare the content of the Numpy array before and after a Parameter is stored."""
        array = np.ma.arange(100)
        param = Parameter('Test', array=array)
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.set_parameter(param)

        with FlightDataFile(self.fp) as fdf:
            param = fdf.get_parameter('Test')
            self.assertTrue(np.all(param.array == array))

    def test_get_parameter_cache(self):
        """
        Get parameter, the retrieved object must be the same every time

        copy_param=False is required for comparison purposes.
        """
        with FlightDataFile(self.fp, cache_param_list=True) as fdf:
            param1 = fdf.get_parameter('Airspeed', copy_param=False)
            self.assertIn(param1.name, fdf.parameter_cache)
            cached = fdf.parameter_cache['Airspeed']
            self.assertEqual(param1, cached)
            param2 = fdf.get_parameter('Airspeed', copy_param=False)
            self.assertEqual(param2, cached)

    def test_get_parameter_valid_only(self):
        """Ensure an exception is raised on acces of invalid parameter when valid_only=True is used."""
        with FlightDataFile(self.fp, mode='a') as fdf:
            pitch = fdf['Airspeed']
            pitch.invalid = True
            fdf['Airspeed'] = pitch

        with FlightDataFile(self.fp) as fdf:
            with self.assertRaises(KeyError):
                fdf.get_parameter('Airspeed', valid_only=True)

    def test_get_parameter_slice(self):
        """Ensure the parameter is cached."""
        with FlightDataFile(self.fp) as fdf:
            airspeed = fdf.get_parameter('Airspeed', _slice=slice(0, 100))
            # array size is correct
            self.assertEqual(airspeed.array.size, 100 * airspeed.frequency)

    def test_get_parameter_slice_cache(self):
        """Ensure the fetched sliced parameter is cached."""
        with FlightDataFile(self.fp) as fdf:
            fdf.cache_param_list += fdf.keys()
            fdf.get_parameter('Airspeed', _slice=slice(0, 100))
            # parameter is cached
            self.assertIn('Airspeed', fdf.parameter_cache)

    def test_get_parameter_load_submasks(self):
        """Ensure the submasks are loaded correctly and cached."""
        with FlightDataFile(self.fp) as fdf:
            airs = fdf.get_parameter('Airspeed', load_submasks=True)
            self.assertItemsEqual(airs.submasks.keys(), json.loads(fdf.data['Airspeed'].attrs['submasks']).keys())

    def test_get_parameter_load_submasks_cache(self):
        """Ensure the submasks are loaded correctly and cached."""
        with FlightDataFile(self.fp) as fdf:
            fdf.cache_param_list += fdf.keys()
            airs = fdf.get_parameter('Airspeed', load_submasks=True)
            self.assertItemsEqual(fdf.parameter_cache['Airspeed'].submasks.keys(), airs.submasks.keys())

    def test_get_parameter_copy(self):
        """Get parameter with copy_param twice, the copies should differ."""
        with FlightDataFile(self.fp) as fdf:
            param1 = fdf.get_parameter('Airspeed', copy_param=True)
            param2 = fdf.get_parameter('Airspeed', copy_param=True)
            self.assertNotEquals(param1, param2)

    def test_set_parameter(self):
        """Set parameter, compare with raw data in the file."""
        array = np.ma.arange(100)
        array.mask = np.zeros(100)
        array.mask[0] = True
        array.mask[-1] = True
        param = Parameter('Test', array=array)
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.set_parameter(param)

        with FlightDataFile(self.fp) as fdf:
            parameter = fdf['Test']
            self.assertTrue(np.all(parameter.array == array))

        with h5py.File(self.fp, mode='r') as hdf:
            data = self.get_data_from_hdf(hdf)['Test']['data']
            self.assertTrue(np.all(param.array == data))

    def test_set_parameter_submasks(self):
        """Set parameter with submasks."""
        array = np.ma.arange(100)
        array.mask = np.zeros(100)
        array.mask[0] = True
        array.mask[-1] = True
        sub1 = np.zeros(100, dtype=np.bool)
        sub1[0] = True
        sub2 = np.zeros(100, dtype=np.bool)
        sub2[-1] = True
        submasks = {'sub1': sub1, 'sub2': sub2}
        param = Parameter('Test', array=array, submasks=submasks)
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.set_parameter(param)

        with h5py.File(self.fp, mode='r') as hdf:
            mask = self.get_data_from_hdf(hdf)['Test']['submasks']
            submasks = json.loads(self.get_data_from_hdf(hdf)['Test'].attrs['submasks'])
            self.assertTrue(np.all(mask[:, submasks['sub1']] == sub1))
            self.assertTrue(np.all(mask[:, submasks['sub2']] == sub2))

    def test_delete_parameter(self):
        """Ensure deleted parameter is not found in the file."""
        with FlightDataFile(self.fp, mode='a') as fdf:
            self.assertIn('Airspeed', fdf)
            fdf.delete_parameter('Airspeed')

        with FlightDataFile(self.fp) as fdf:
            self.assertNotIn('Airspeed', fdf)

    def test_get_parameters(self):
        """Ensure all requested parameters are returned."""
        parameter_names = ('Airspeed', 'Heading')
        with FlightDataFile(self.fp) as fdf:
            parameters = fdf.get_parameters(parameter_names)

        self.assertItemsEqual(parameter_names, (p for p in parameters))

    def test_set_parameters(self):
        """Ensure all stored parameters are found in the file."""
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(100)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.set_parameters(params)

        with FlightDataFile(self.fp) as fdf:
            for name in parameter_names:
                self.assertIn(name, fdf)

    def test_delete_parameters(self):
        """Ensure all deleted parameters are not found in the file."""
        parameter_names = ('Airspeed', 'Heading')
        with FlightDataFile(self.fp, mode='a') as fdf:
            fdf.delete_parameters(parameter_names)

        with FlightDataFile(self.fp) as fdf:
            for name in parameter_names:
                self.assertNotIn(name, fdf)

    def test_parameter_cache(self):
        """Ensure cache is populated on parameter access."""
        with FlightDataFile(self.fp) as fdf:
            fdf.cache_param_list += fdf.keys()
            # cache empty
            self.assertFalse(fdf.parameter_cache)
            # access a parameter
            fdf['Airspeed']
            self.assertIn('Airspeed', fdf.parameter_cache)

    def test_trim_full(self):
        """Trim the file without limit (full copy)."""
        with FlightDataFile(self.fp) as fdf:
            duration = fdf.duration
            fdf.trim(self.fp + '-trim')

        with FlightDataFile(self.fp + '-trim') as fdf:
            self.assertEqual(fdf.duration, duration)
            for parameter in fdf.values():
                self.assertEqual(len(parameter.array), math.floor(duration * parameter.frequency))

    def test_trim_slice(self):
        """Trim the file to first 100 seconds."""
        with FlightDataFile(self.fp) as fdf:
            fdf.trim(self.fp + '-trim', stop_offset=100, superframe_boundary=False)

        with FlightDataFile(self.fp + '-trim') as fdf:
            # XXX: the test data has superframe parameters, trimming it without superframe boundary will lead to
            # inconsistent number of samples and inconsistent duration!
            for parameter in fdf.values():
                if parameter.frequency < 0.25:
                    continue
                self.assertEqual(parameter.duration, 100)
                self.assertEqual(len(parameter.array), math.floor(100 * parameter.frequency))

    def test_trim_slice_superframe_boundary(self):
        """Trim the file to first 100 seconds aligned to superframes."""
        with FlightDataFile(self.fp) as fdf:
            fdf.trim(self.fp + '-trim', stop_offset=100, superframe_boundary=True)

        with FlightDataFile(self.fp + '-trim') as fdf:
            self.assertEqual(fdf.duration, 128)
            for parameter in fdf.values():
                self.assertEqual(len(parameter.array), 128 * parameter.frequency)


class FlightDataFileTestV3(FlightDataFileTestV2):
    test_fn = 'data/flight_data_v3.hdf5'

    def get_data_from_hdf(self, hdf):
        '''Return the object storing parameters (depending on version)'''
        return hdf
