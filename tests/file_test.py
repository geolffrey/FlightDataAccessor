import h5py
import mock
import numpy as np
import os
import random
import simplejson
import unittest

from flightdataaccessor.file import hdf_file
from flightdataaccessor.datatypes.parameter import Parameter

TEST_DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
TEMP_DIR_PATH = os.path.join(TEST_DATA_DIR_PATH, 'temp')
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')


class TestHdfFile(unittest.TestCase):

    def setUp(self):
        self.hdf_path = os.path.join(TEST_DATA_DIR, 'test_hdf_access.hdf')
        with h5py.File(self.hdf_path, 'w') as hdf:
            series = hdf.create_group('series')
            self.param_name = 'TEST_PARAM10'
            param_group = series.create_group(self.param_name)
            self.param_frequency = 2
            self.param_supf_offset = 1.5
            self.param_arinc_429 = True
            param_group.attrs['frequency'] = self.param_frequency
            param_group.attrs['supf_offset'] = self.param_supf_offset
            param_group.attrs['arinc_429'] = self.param_arinc_429
            param_group.attrs['lfl'] = 1
            param_group.attrs['invalid'] = 0
            self.param_data = np.arange(100)
            param_group.create_dataset('data', data=self.param_data)
            self.masked_param_name = 'TEST_PARAM11'
            masked_param_group = series.create_group(self.masked_param_name)
            self.masked_param_frequency = 4
            self.masked_param_supf_offset = 2.5
            masked_param_group.attrs['frequency'] = self.masked_param_frequency
            masked_param_group.attrs['supf_offset'] = self.masked_param_supf_offset
            self.param_mask = [bool(random.randint(0, 1)) for x in range(len(self.param_data))]
            masked_param_group.create_dataset('data', data=self.param_data)
            masked_param_group.create_dataset('mask', data=self.param_mask)
            self.masked_param_submask_arrays = np.column_stack([self.param_mask, self.param_mask])
            self.masked_param_submask_map = {'mask1': 0, 'mask2': 1}
            masked_param_group.attrs['submasks'] = \
                simplejson.dumps(self.masked_param_submask_map)
            masked_param_group.create_dataset(
                'submasks', data=self.masked_param_submask_arrays)
            masked_param_group.attrs['lfl'] = 0
            masked_param_group.attrs['invalid'] = 0

    def tearDown(self):
        try:
            os.remove(self.hdf_path)
        except (IOError, OSError):
            pass

    def test_dependency_tree(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            self.assertEqual(fdf.dependency_tree, None)
            dependency_tree = {'Airspeed': ['Altitude AAL'], 'Altitude AAL': []}
            fdf.dependency_tree = dependency_tree
            self.assertEqual(fdf.dependency_tree, dependency_tree)
            fdf.dependency_tree = None
            self.assertEqual(fdf.dependency_tree, None)

    def test_get_matching(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            regex_str = '^TEST_PARAM10$'
            params = fdf.get_matching(regex_str)
            self.assertEqual(len(params), 1)
            self.assertEqual(params[0].name, self.param_name)
            regex_str = '^TEST_PARAM1[01]$'
            params = fdf.get_matching(regex_str)
            self.assertEqual(len(params), 2)
            param_names = [p.name for p in params]
            self.assertTrue(self.param_name in param_names)
            self.assertTrue(self.masked_param_name in param_names)

    def test_open_and_close_and_full_masks(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            # check it's open
            self.assertFalse(fdf.hdf.id is None)
            fdf['sample'] = Parameter('sample', np.array(range(10)))
            np.testing.assert_almost_equal(fdf['sample'].array.data, list(range(10)))
            self.assertTrue(hasattr(fdf['sample'].array, 'mask'))

            fdf['masked sample'] = Parameter('masked sample', np.ma.array(range(10)))
            self.assertEqual(list(fdf['masked sample'].array.data), list(range(10)))
            # check masks are returned in full (not just a single False)
            self.assertEqual(list(fdf['masked sample'].array.mask), [False] * 10)

        # check it's closed
        self.assertIsNone(fdf.hdf)

    def test_limit_storage(self):
        # test set and get param limits
        parameter_limits = {
            'TEST_PARAM10': {
                'arinc': None,
                'rate_of_change_limit': None,
                'max_limit': 41000,
                'min_limit': -1000
            }
        }

        with hdf_file(self.hdf_path, mode='a') as hdf:
            hdf.set_param_limits(self.param_name, parameter_limits.get('TEST_PARAM10'))

            stored_param_limits = hdf.get_param_limits(self.param_name)
            self.assertEqual(parameter_limits.get('TEST_PARAM10'), stored_param_limits)

    def test_get_params(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            # Test retrieving all parameters.
            params = fdf.get_params()
            self.assertTrue(len(params) == 2)
            param = params['TEST_PARAM10']
            self.assertEqual(param.frequency, self.param_frequency)
            self.assertEqual(param.arinc_429, self.param_arinc_429)
            param = params['TEST_PARAM11']
            self.assertEqual(param.offset, self.masked_param_supf_offset)
            self.assertEqual(param.arinc_429, False)
            # Test retrieving single specified parameter.
            params = fdf.get_params(param_names=['TEST_PARAM10'])
            self.assertTrue(len(params) == 1)
            param = params['TEST_PARAM10']
            self.assertEqual(param.frequency, self.param_frequency)

    def test_get_param_valid_only(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            fdf['Valid Param'] = Parameter('Valid Param', array=np.ma.arange(10))
            fdf['Invalid Param'] = Parameter('Invalid Param', array=np.ma.arange(10), invalid=True)
            self.assertIn('Invalid Param', fdf)
            # request Invalid param without filtering
            self.assertTrue(fdf.get_param('Invalid Param', valid_only=False))
            # filtering only valid raises keyerror
            self.assertRaises(KeyError, fdf.get_param, 'Invalid Param', valid_only=True)

    def test_get_params_valid_only(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            fdf['Valid Param'] = Parameter('Valid Param', array=np.ma.arange(10))
            fdf['Invalid Param'] = Parameter('Invalid Param', array=np.ma.arange(10), invalid=True)
            self.assertIn('Invalid Param', fdf)
            # check the params that are valid are listed correctly
            self.assertEqual(fdf.valid_param_names(), ['TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
            # request all params inc. invalid
            all_params = fdf.get_params(valid_only=False)
            self.assertEqual(
                sorted(all_params.keys()), ['Invalid Param', 'TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
            # request only valid params
            valid = fdf.get_params(valid_only=True)
            self.assertEqual(sorted(valid.keys()), ['TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
            # request params by name
            valid_few = fdf.get_params(param_names=['Valid Param'], valid_only=True)
            self.assertEqual(list(valid_few.keys()), ['Valid Param'])
            # try to request the invalid param, but only accepting valid raises keyerror
            self.assertRaises(
                KeyError, fdf.get_params, param_names=['Invalid Param', 'Valid Param'], valid_only=True,
                raise_keyerror=True)

    def test___set_item__(self):
        '''
        set_item uses set_param
        '''
        with hdf_file(self.hdf_path, mode='a') as fdf:
            set_param_data = fdf.__setitem__
            # Create new parameter with np.array.
            name1 = 'TEST_PARAM1'
            array = np.arange(100, dtype=np.float_)
            set_param_data(name1, Parameter(name1, array))
            self.assertTrue(np.all(fdf.data[name1]['data'][:] == array))
            self.assertFalse('arinc_429' in fdf.data[name1].attrs)
            # Create new parameter with np.ma.masked_array.
            name2 = 'TEST_PARAM2'
            mask = [False] * len(array)
            masked_array = np.ma.masked_array(data=array, mask=mask)
            param2_frequency = 0.5
            param2_offset = 2
            param2_arinc_429 = True
            set_param_data(name2, Parameter(name2, masked_array,
                                            frequency=param2_frequency,
                                            offset=param2_offset,
                                            arinc_429=param2_arinc_429))
            self.assertTrue(np.all(fdf.data[name2]['data'].value == array))
            self.assertEqual(fdf.data[name2].attrs['frequency'], param2_frequency)
            self.assertEqual(fdf.data[name2].attrs['offset'], param2_offset)
            self.assertEqual(fdf.data[name2].attrs['arinc_429'], param2_arinc_429)

            # Set existing parameter's data with np.array.
            array = np.arange(200)
            set_param_data(name1, Parameter(name1, array))
            self.assertTrue(np.all(fdf.data[name1]['data'].value == array))

            # Set existing parameter's data with np.ma.masked_array.
            mask = [bool(random.randint(0, 1)) for x in range(len(array))]
            masked_array = np.ma.masked_array(data=array, mask=mask)
            set_param_data(name1, Parameter(name1, masked_array))
            self.assertTrue(np.all(fdf.data[name1]['data'].value == array))

            # Save submasks.
            array = np.ma.arange(100)
            array.mask = np.zeros(len(array)).astype(np.bool)
            array.mask[:3] = [True, True, False]
            mask1 = np.zeros(len(array)).astype(np.bool)
            mask1[:3] = [False, True, False]
            mask2 = np.zeros(len(array)).astype(np.bool)
            mask2[:3] = [True, False, False]
            submasks = {'mask1': mask1, 'mask2': mask2}
            set_param_data(name1, Parameter(name1, array, submasks=submasks))
            submask_map = simplejson.loads(fdf.data[name1].attrs['submasks'])
            submask_arrays = fdf.data[name1]['submasks'][:]
            self.assertEqual(submasks['mask1'].tolist(), submask_arrays[:, submask_map['mask1']].tolist())
            self.assertEqual(submasks['mask2'].tolist(), submask_arrays[:, submask_map['mask2']].tolist())

    def test_update_param_mask(self):
        # setup original array
        name1 = 'Airspeed Minus Vref'
        array = np.arange(200)
        with hdf_file(self.hdf_path, mode='a') as fdf:
            fdf.set_param(Parameter(name1, array))
            series = fdf.hdf['series']

            # Set mask changes, but not Data array
            new_array = np.arange(200)[::-1]
            new_mask = [bool(random.randint(0, 1)) for x in range(len(new_array))]
            masked_array = np.ma.masked_array(data=new_array, mask=new_mask)

            fdf.set_param(Parameter(name1, masked_array), save_data=False, save_mask=True)
            # asssert data remains same as original array
            self.assertTrue(np.all(series[name1]['data'].value == array))

    def test_update_param_attributes(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            # save initial Parameter to file
            fdf.set_param(Parameter('Blah', np.ma.arange(1)))
            # Update the invalidity flag only
            fdf.set_param(Parameter('Blah', np.ma.arange(1), invalid=1), save_data=False, save_mask=False)
            self.assertEqual(fdf['Blah'].invalid, 1)

    def test_get_param_data(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            self.__test_get_param_data(fdf.get_param)
            param = fdf.get_param(self.masked_param_name, load_submasks=True)
            self.assertEqual(self.masked_param_submask_map.keys(), param.submasks.keys())
            self.assertEqual(
                self.masked_param_submask_arrays[:, self.masked_param_submask_map['mask1']].tolist(),
                param.submasks['mask1'].tolist())
            self.assertEqual(
                self.masked_param_submask_arrays[:, self.masked_param_submask_map['mask2']].tolist(),
                param.submasks['mask2'].tolist())

    def test___get_item__(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            self.__test_get_param_data(fdf.__getitem__)

    def __test_get_param_data(self, get_param):
        '''
        :param set_param_data: Allows passing of either hdf_file.get_param or __getitem__.
        :type get_param_data: method
        '''
        # Create new parameter with np.array.
        param = get_param(self.param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertEqual(self.param_frequency, param.frequency)
        self.assertEqual(self.param_supf_offset, param.offset)
        # Create new parameter with np.array.
        param = get_param(self.masked_param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertTrue(np.all(self.param_mask == param.array.mask))
        self.assertEqual(self.masked_param_frequency, param.frequency)
        self.assertEqual(self.masked_param_supf_offset, param.offset)
        # XXX: 'derived' submask is created from the parameter's mask!
        self.assertItemsEqual(['derived'], param.submasks.keys())
        np.testing.assert_array_equal(param.submasks['derived'], np.ma.getmaskarray(param.array))

    def test_len(self):
        '''
        Depends upon HDF creation in self.setUp().
        '''
        with hdf_file(self.hdf_path) as fdf:
            self.assertEqual(len(fdf), 2)

    def test_keys(self):
        with hdf_file(self.hdf_path) as fdf:
            self.assertEqual(sorted(fdf.keys()), sorted([self.param_name, self.masked_param_name]))

    def test_lfl_keys(self):
        with hdf_file(self.hdf_path) as fdf:
            print fdf.lfl_keys(), fdf.derived_keys(), fdf.keys()
            self.assertEqual(sorted(fdf.lfl_keys()), ['TEST_PARAM10'])

    def test_derived_keys(self):
        with hdf_file(self.hdf_path) as fdf:
            self.assertEqual(sorted(fdf.derived_keys()), ['TEST_PARAM11'])

    def test_startswith(self):
        with hdf_file(self.hdf_path) as fdf:
            self.assertEqual(fdf.startswith('TEST_'), ['TEST_PARAM10', 'TEST_PARAM11'])

    @unittest.skip('Breaks compatibility with dynamic attributes (overwrites the parameter list)')
    def test_search(self):
        params = ['ILS Localizer', 'ILS Localizer (R)', 'ILS Localizer (L)', 'Rate of Climb', 'Altitude STD',
                  'Brake (R) Pressure Ourboard', 'Brake (L) Pressure Inboard', 'ILS Localizer Deviation Warning',
                  'ILS Localizer Test Tube Inhibit', 'ILS Localizer Beam Anomaly', 'ILS Localizer Engaged']

        mock_keys = mock.Mock(spec=['keys'], return_value=params)
        with hdf_file(self.hdf_path) as fdf:
            object.__setattr__(fdf, 'keys', mock_keys)

            search_key = 'ILS Localizer'

            expected_output = [
                'ILS Localizer', 'ILS Localizer (L)', 'ILS Localizer (R)', 'ILS Localizer Beam Anomaly',
                'ILS Localizer Deviation Warning', 'ILS Localizer Engaged', 'ILS Localizer Test Tube Inhibit']

            res = fdf.search(search_key)
            self.assertEqual(res, expected_output)

            search_key_star = 'ILS Localizer (*)'

            expected_output_star = ['ILS Localizer', 'ILS Localizer (L)', 'ILS Localizer (R)']
            res = fdf.search(search_key_star)
            self.assertEqual(res, expected_output_star)

    def test_mapped_array(self):
        # created mapped array
        mapping = {0: 'zero', 2: 'two', 3: 'three'}
        array = np.ma.array(list(range(5)) + list(range(5)), mask=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        multi_p = Parameter('multi', array, values_mapping=mapping)

        # save array to hdf
        with hdf_file(self.hdf_path, mode='a') as fdf:
            fdf['multi'] = multi_p

        # check hdf has mapping and integer values stored
        with hdf_file(self.hdf_path, mode='a') as fdf:
            multi_p = fdf['multi']
            self.assertEqual(str(multi_p.array[:]), "[-- -- -- 'three' '?' 'zero' '?' 'two' -- --]")
            self.assertEqual(multi_p.array.data.dtype, np.int)
            # modify a masked item
            multi_p.array[0] = 'three'
            with self.assertRaises(ValueError):
                # XXX this fails because we've changed the mask by assignment above but we've not updated the submask
                fdf['multi'] = multi_p

    def test__delitem__(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            p = 'TEST_PARAM10'
            self.assertTrue(p in fdf)
            del fdf[p]
            self.assertFalse(p in fdf)
            self.assertFalse(p in fdf.hdf['series'])

            self.assertRaises(KeyError, fdf.__delitem__, 'INVALID_PARAM_NAME')

    def test_delete_params(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            ps = ['TEST_PARAM10', 'TEST_PARAM11', 'INVALID_PARAM_NAME']
            self.assertTrue(ps[0] in fdf)
            self.assertTrue(ps[1] in fdf)
            self.assertFalse(ps[2] in fdf)  # invalid is not there
            # delete and ensure by default keyerrors are supressed
            fdf.delete_params(ps)
            for p in ps:
                self.assertFalse(p in fdf)

    def test_create_file(self):
        temp = 'temp_new_file.hdf5'
        if os.path.exists(temp):
            os.remove(temp)
        # cannot create file without specifying 'create=True'
        # self.assertRaises(IOError, hdf_file, temp)
        self.assertFalse(os.path.exists(temp))
        # this one will create the file
        hdf = hdf_file(temp, create=True)
        self.assertTrue(os.path.exists(temp))
        self.assertEqual(hdf.version, hdf.VERSION)
        os.remove(temp)

    def test_set_and_get_attributes(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            # Test setting a datetime as it's a non-json non-string type.
            self.assertFalse(fdf.hdf.attrs.get('start_datetime'))
            fdf.set_attr('reloable_frame_counter', False)
            self.assertEqual(fdf.get_attr('non-existing'), None)
            self.assertEqual(fdf.get_attr('non-existing', default='default'), 'default')

    def test_set_invalid(self):
        with hdf_file(self.hdf_path, mode='a') as fdf:
            name = 'TEST_PARAM11'
            fdf.set_invalid(name)
            p = fdf[name]
            self.assertTrue(p.invalid)
            self.assertEqual(p.invalidity_reason, '')
            # XXX: v3 API does not set the mask
            # self.assertTrue(p.array.mask.all())
            fdf.set_invalid(name, 'Unplugged')
            p = fdf[name]
            self.assertTrue(p.invalid)
            self.assertEqual(p.invalidity_reason, 'Unplugged')
            # XXX: v3 API does not set the mask
            # self.assertTrue(p.array.mask.all())
            self.assertRaises(KeyError, fdf.set_invalid, 'WRONG_NAME')
