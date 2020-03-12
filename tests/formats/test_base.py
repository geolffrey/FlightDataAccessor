import math
import unittest

import numpy as np

from flightdataaccessor import FlightDataFormat, Parameter


class FlightDataFormatTest(unittest.TestCase):
    def assertItemsEqual(self, l1, l2):
        self.assertEqual(set(l1), set(l2))

    def test_setitem(self):
        """__setitem__ is equivalent to set_parameter()"""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)

    def test_getitem(self):
        """Ensure the name of retrieved parameter is correct."""
        param = Parameter('Test', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param = fdf['Test']
            self.assertEqual(param.name, 'Test')
            with self.assertRaises(KeyError):
                fdf['Missing']

    def test_delitem(self):
        """__delitem__ is equivalent to delete_parameter()"""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)
            del fdf['Test']
            with self.assertRaises(KeyError):
                del fdf['Test']
            self.assertNotIn('Test', fdf)

    def test_contains(self):
        """Ensure `in` operator works for the object."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertTrue('Test' in fdf)
            self.assertFalse('Nonexistent' in fdf)

    def test_len(self):
        """Compare len of the FlightDataFormat object with the len of data series in raw HDF file."""
        with FlightDataFormat() as fdf:
            self.assertTrue(len(fdf) == len(fdf.data.keys()))

    def test_keys(self):
        """Test the consistency of keys cache for all parameters."""
        with FlightDataFormat() as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(derived_param_names + lfl_param_names, fdf.keys())

    def test_keys_valid_only(self):
        """Test the consistency of keys cache for valid parameters."""
        with FlightDataFormat() as fdf:
            valid_param_names = (p.name for p in fdf.values() if not p.invalid)
            self.assertItemsEqual(valid_param_names, fdf.keys(valid_only=True))

            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source]
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and not p.source]
            self.assertItemsEqual(derived_valid_param_names + source_valid_param_names, fdf.keys(valid_only=True))

    def test_keys_source(self):
        """Test the consistency of keys cache for source parameters."""
        with FlightDataFormat() as fdf:
            lfl_param_names = [p.name for p in fdf.values() if p.source == 'lfl']
            self.assertItemsEqual(lfl_param_names, fdf.keys(subset='lfl'))

    def test_keys_derived(self):
        """Test the consistency of keys cache for derived parameters."""
        with FlightDataFormat() as fdf:
            derived_param_names = [p.name for p in fdf.values() if p.source == 'derived']
            self.assertItemsEqual(derived_param_names, fdf.keys(subset='derived'))

    def test_keys_valid_source(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFormat() as fdf:
            source_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'lfl']
            self.assertItemsEqual(source_valid_param_names, fdf.keys(valid_only=True, subset='lfl'))

    def test_keys_valid_derived(self):
        """Test the consistency of keys cache for valid source parameters."""
        with FlightDataFormat() as fdf:
            derived_valid_param_names = [p.name for p in fdf.values() if not p.invalid and p.source == 'derived']
            self.assertItemsEqual(derived_valid_param_names, fdf.keys(valid_only=True, subset='derived'))

    def test_keys_add(self):
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

    def test_values(self):
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

    def test_items(self):
        """
        Compare parameter names extracted with values() before and after adding a parameter.
        """
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', array=np.ma.arange(100))
        with FlightDataFormat() as fdf:
            fdf.set_parameters((param1, param2))
            param_names = set()
            for param_name, param in fdf.items():
                self.assertEqual(param_name, param.name)
                param_names.add(param_name)

    def test_duration(self):
        """Verify that file duration is calculated automatically on access."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertEqual(fdf.duration, 200)
            fdf['Test'] = param.trim(0, 100)
            self.assertEqual(fdf.duration, 100)

    def test_frequencies(self):
        """Verify that file frequencies is calculated automatically on access."""
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', frequency=2, array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            np.testing.assert_array_equal(sorted(fdf.frequencies), [])
            fdf['Test 1'] = param1
            np.testing.assert_array_equal(sorted(fdf.frequencies), [param1.frequency])
            fdf['Test 2'] = param2
            np.testing.assert_array_equal(sorted(fdf.frequencies), sorted([param1.frequency, param2.frequency]))

    def test_get_parameter(self):
        """Compare the content of the Numpy array before and after a Parameter is stored."""
        array = np.ma.arange(100)
        param = Parameter('Test', array=array)
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param)
            param = fdf.get_parameter('Test')
            self.assertTrue(np.all(param.array == array))

    def test_get_parameter_valid_only(self):
        """Ensure an exception is raised on acces of invalid parameter when valid_only=True is used."""
        param1 = Parameter('Test 1', array=np.ma.arange(200))
        param2 = Parameter('Test 2', array=np.ma.arange(200), invalid=True)
        with FlightDataFormat() as fdf:
            fdf['Test 1'] = param1
            fdf['Test 2'] = param2
            with self.assertRaises(KeyError):
                fdf.get_parameter('Test 2', valid_only=True)

    def test_get_parameter_slice(self):
        """Ensure the parameter is cached."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param = fdf.get_parameter('Test', _slice=slice(0, 100))
            self.assertEqual(param.array.size, 100 * param.frequency)

    def test_get_parameter_copy(self):
        """Get parameter with copy_param twice, the copies should differ."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            param1 = fdf.get_parameter('Test', copy_param=False)
            param2 = fdf.get_parameter('Test', copy_param=False)
            self.assertEqual(param1, param2)

            param1 = fdf.get_parameter('Test', copy_param=True)
            param2 = fdf.get_parameter('Test', copy_param=True)
            self.assertNotEquals(param1, param2)

    def test_set_parameter(self):
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
        with FlightDataFormat() as fdf:
            fdf.set_parameter(param)

    def test_delete_parameter(self):
        """Ensure deleted parameter is not found in the file."""
        array = np.ma.arange(100)
        param = Parameter('Test', array=array)
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            self.assertIn('Test', fdf)
            fdf.delete_parameter('Test')
            self.assertNotIn('Test', fdf)

    def test_get_parameters(self):
        """Ensure all requested parameters are returned."""
        param1 = Parameter('Test 1', array=np.ma.arange(100))
        param2 = Parameter('Test 2', frequency=2, array=np.ma.arange(200))
        parameter_names = ('Test 1', 'Test 2')
        with FlightDataFormat() as fdf:
            fdf.set_parameters((param1, param2))
            parameters = fdf.get_parameters(parameter_names)

        self.assertItemsEqual(parameter_names, (p for p in parameters))

    def test_set_parameters(self):
        """Ensure all stored parameters are found in the file."""
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(100)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFormat() as fdf:
            fdf.set_parameters(params)
            for name in parameter_names:
                self.assertIn(name, fdf)

    def test_delete_parameters(self):
        """Ensure all deleted parameters are not found in the file."""
        parameter_names = ('Test 1', 'Test 2')
        array = np.ma.arange(100)
        params = (Parameter(name, array=array) for name in parameter_names)
        with FlightDataFormat() as fdf:
            fdf.set_parameters(params)
            fdf.delete_parameters(parameter_names)
            for name in parameter_names:
                self.assertNotIn(name, fdf)

    def test_trim_deidentify(self):
        keep_parameters = (
            'Airspeed',
            'Eng (1) N1',
        )
        remove_parameters = (
            'AC Ident',
            'AC Number',
            'AC Tail',
            'AC Type',
            'ACMS Software PN Code',
            'ACMS Software Part Number Code',
            'Date',
            'Eng Aircraft Type Code',
            'Eng (1) Aircraft Type Code',
            'Eng Model',
            'Eng (2) Model',
            'Fleet Number',
            'Flight Number',
            'Manufacturer Code',
            'Serial Number',
        )

        data = np.ma.arange(8)
        with FlightDataFormat() as fdf:
            for parameter_name in keep_parameters + remove_parameters:
                fdf.set_parameter(Parameter(parameter_name, array=data.copy()))
            fdf.set_parameter(Parameter('Day', array=np.ma.array([5,5,5,6,6,6,6,6])))

            trimmed = fdf.trim(deidentify=True)

            for parameter_name in keep_parameters:
                self.assertTrue(parameter_name in trimmed)
                self.assertTrue(np.ma.all(trimmed[parameter_name].array == data))

            for parameter_name in remove_parameters:
                self.assertFalse(parameter_name in trimmed)

            self.assertTrue('Day' in trimmed)
            self.assertTrue(np.ma.all(trimmed['Day'].array == np.ma.array([1,1,1,2,2,2,2,2])))

    def test_trim_full(self):
        """Trim the file to first 100 seconds."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            trimmed = fdf.trim()
            self.assertEqual(fdf.duration, trimmed.duration)
            for parameter in fdf.values():
                self.assertEqual(len(parameter.array), math.floor(fdf.duration * parameter.frequency))

    def test_trim_slice(self):
        """Trim the file to first 100 seconds."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            trimmed = fdf.trim(stop_offset=100)
            self.assertEqual(trimmed.duration, 100)
            for parameter in trimmed.values():
                self.assertEqual(len(parameter.array), math.floor(100 * parameter.frequency))

    def test_trim_slice_superframe_boundary(self):
        """Trim the file to first 100 seconds aligned to superframes."""
        param = Parameter('Test', array=np.ma.arange(200))
        with FlightDataFormat() as fdf:
            fdf['Test'] = param
            fdf.superframe_present = True
            trimmed = fdf.trim(stop_offset=100, superframe_boundary=True)
            self.assertEqual(trimmed.duration, 128)
            for parameter in trimmed.values():
                self.assertEqual(len(parameter.array), 128 * parameter.frequency)
