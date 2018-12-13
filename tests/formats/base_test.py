import math
import unittest

import numpy as np

from flightdataaccessor.formats.base import FlightDataFormat
from flightdataaccessor.datatypes.parameter import Parameter


class FlightDataFormatTest(unittest.TestCase):
    def assertItemsEqual(self, l1, l2):
        self.assertEqual(set(l1), set(l2))

    def setitem_test(self):
        """__setitem__ is equivalent to set_parameter()"""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)

    def getitem_test(self):
        """Ensure the name of retrieved parameter is correct."""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param = fdf['Test']
            self.assertEqual(param.name, 'Test')

    def delitem_test(self):
        """__delitem__ is equivalent to delete_parameter()"""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)
            del fdf['Test']
            with self.assertRaises(KeyError):
                del fdf['Test']
            self.assertNotIn('Test', fdf)

    def contains_test(self):
        """Ensure `in` operator works for the object."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertTrue('Test' in fdf)
            self.assertFalse('Nonexistent' in fdf)

    def len_test(self):
        """Compare len of the FlightDataFormat object with the len of data series in raw HDF file."""
        with FlightDataFormat() as fdf:
            self.assertTrue(len(fdf) == len(fdf.data.keys()))

    def keys_test(self):
        """Test the consistency of keys cache for all parameters."""
        with FlightDataFormat() as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(derived_param_names + lfl_param_names, fdf.keys())

    def keys_valid_only_test(self):
        """Test the consistency of keys cache for valid parameters."""
        with FlightDataFormat() as fdf:
            valid_param_names = (p.name for p in fdf.values() if not p.invalid)
            self.assertItemsEqual(valid_param_names, fdf.keys(valid_only=True))

            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source]
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and not p.source]
            self.assertItemsEqual(derived_valid_param_names + source_valid_param_names, fdf.keys(valid_only=True))

    def keys_source_test(self):
        """Test the consistency of keys cache for source parameters."""
        with FlightDataFormat() as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            self.assertItemsEqual(lfl_param_names, fdf.keys(subset='lfl'))

    def keys_derived_test(self):
        """Test the consistency of keys cache for derived parameters."""
        with FlightDataFormat() as fdf:
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(derived_param_names, fdf.keys(subset='derived'))

    def keys_valid_source_test(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFormat() as fdf:
            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'lfl']
            self.assertItemsEqual(source_valid_param_names, fdf.keys(valid_only=True, subset='lfl'))

    def keys_valid_derived_test(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFormat() as fdf:
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'derived']
            self.assertItemsEqual(derived_valid_param_names, fdf.keys(valid_only=True, subset='derived'))

    def keys_add_test(self):
        """Compare keys() before and after adding a parameter."""
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param1)
            old_keys = fdf.keys()
            self.assertNotIn('Test 2', old_keys)
            fdf.set_parameter(param2)
            new_keys = fdf.keys()
            self.assertIn('Test 2', new_keys)

    def values_test(self):
        """
        Compare parameter names extracted with values() before and after adding a parameter.
        """
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf.set_parameters((param1, param2))
            fdf_param_names = (p.name for p in fdf.values())
            self.assertItemsEqual(fdf_param_names, ['Test 1', 'Test 2'])

            param = Parameter('Test 3', array=np.ma.arange(100))
            self.assertNotIn(param.name, fdf_param_names)
            fdf.set_parameter(param)
            fdf_param_names = (p.name for p in fdf.values())
            self.assertItemsEqual(fdf_param_names, ['Test 1', 'Test 2', 'Test 3'])

    def items_test(self):
        """
        Compare parameter names extracted with values() before and after adding a parameter.
        """
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf.set_parameters((param1, param2))
            param_names = set()
            for param_name, param in fdf.items():
                self.assertEquals(param_name, param.name)
                param_names.add(param_name)

    def duration_test(self):
        """Verify that file duration is calculated automatically on access."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertEquals(fdf.duration, 200)
            fdf['Test'] = param.trim(0, 100)
            self.assertEquals(fdf.duration, 100)

    def frequencies_test(self):
        """Verify that file frequencies is calculated automatically on access."""
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', frequency=2, array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            np.testing.assert_array_equal(sorted(fdf.frequencies), [])
            fdf['Test 1'] = param1
            np.testing.assert_array_equal(sorted(fdf.frequencies), [param1.frequency])
            fdf['Test 2'] = param2
            np.testing.assert_array_equal(sorted(fdf.frequencies), sorted([param1.frequency, param2.frequency]))

    def get_parameter_test(self):
        """Compare the content of the Numpy array before and after a Parameter is stored."""
        array = np.ma.arange(100)
        param = Parameter('Test', array=array)
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param)
            param = fdf.get_parameter('Test')
            self.assertTrue(np.all(param.array == array))

    def get_parameter_valid_only_test(self):
        """Ensure an exception is raised on acces of invalid parameter when valid_only=True is used."""
        param1 = Parameter('Test 1', array=np.ma.arange(200))
        param2 = Parameter('Test 2', array=np.ma.arange(200), invalid=True)
        with FlightDataFormat() as fdf:
            fdf['Test 1'] = param1
            fdf['Test 2'] = param2
            with self.assertRaises(KeyError):
                fdf.get_parameter('Test 2', valid_only=True)

    def get_parameter_slice_test(self):
        """Ensure the parameter is cached."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param = fdf.get_parameter('Test', _slice=slice(0, 100))
            self.assertEquals(param.array.size, 100 * param.frequency)

    def get_parameter_copy_test(self):
        """Get parameter with copy_param twice, the copies should differ."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param1 = fdf.get_parameter('Test', copy_param=False)
            param2 = fdf.get_parameter('Test', copy_param=False)
            self.assertEquals(param1, param2)

            param1 = fdf.get_parameter('Test', copy_param=True)
            param2 = fdf.get_parameter('Test', copy_param=True)
            self.assertNotEquals(param1, param2)

    def set_parameter_test(self):
        """Set parameter, compare with raw data in the file."""
        array = np.ma.arange(100)
        array.mask = np.zeros(100)
        array.mask[0] = True
        array.mask[-1] = True
        param = Parameter('Test', array=array)
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param)
            parameter = fdf['Test']
            self.assertTrue(np.all(parameter.array == array))

    def set_parameter_submasks_test(self):
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
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param)

    def delete_parameter_test(self):
        """Ensure deleted parameter is not found in the file."""
        array = np.ma.arange(100)
        param = Parameter('Test', array=array)
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)
            fdf.delete_parameter('Test')
            self.assertNotIn('Test', fdf)

    def get_parameters_test(self):
        """Ensure all requested parameters are returned."""
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', frequency=2, array=np.ma.arange(200))
        parameter_names = ('Test 1', 'Test 2')
        with FlightDataFormat() as fdf:
            fdf.set_parameters((param1, param2))
            parameters = fdf.get_parameters(parameter_names)

        self.assertItemsEqual(parameter_names, (p for p in parameters))

    def set_parameters_test(self):
        """Ensure all stored parameters are found in the file."""
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(100)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFormat() as fdf:
            fdf.set_parameters(params)
            for name in parameter_names:
                self.assertIn(name, fdf)

    def delete_parameters_test(self):
        """Ensure all deleted parameters are not found in the file."""
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(100)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFormat() as fdf:
            fdf.set_parameters(params)
            fdf.delete_parameters(parameter_names)
            for name in parameter_names:
                self.assertNotIn(name, fdf)

    def trim_full_test(self):
        """Trim the file to first 100 seconds."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            trimmed = fdf.trim()
            self.assertEquals(fdf.duration, trimmed.duration)
            for parameter in fdf.values():
                self.assertEquals(len(parameter.array), math.floor(fdf.duration * parameter.frequency))

    def trim_slice_test(self):
        """Trim the file to first 100 seconds."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            trimmed = fdf.trim(stop_offset=100)
            self.assertEquals(trimmed.duration, 100)
            for parameter in trimmed.values():
                self.assertEquals(len(parameter.array), math.floor(100 * parameter.frequency))

    def trim_slice_superframe_boundary_test(self):
        """Trim the file to first 100 seconds aligned to superframes."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            fdf.superframe_present = True
            trimmed = fdf.trim(stop_offset=100, superframe_boundary=True)
            self.assertEquals(trimmed.duration, 128)
            for parameter in trimmed.values():
                self.assertEquals(len(parameter.array), 128 * parameter.frequency)
