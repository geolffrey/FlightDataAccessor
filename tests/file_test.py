import h5py
import mock
import numpy as np
import os
import random
import calendar
import unittest

from datetime import datetime

from hdfaccess.file import hdf_file
from hdfaccess.parameter import Parameter

TEST_DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
TEMP_DIR_PATH = os.path.join(TEST_DATA_DIR_PATH, 'temp')
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')


class TestHdfFile(unittest.TestCase):

    def setUp(self):
        self.hdf_path = os.path.join(TEST_DATA_DIR, 'test_hdf_access.hdf')
        hdf = h5py.File(self.hdf_path, 'w')
        series = hdf.create_group('series')
        self.param_name = 'TEST_PARAM10'
        param_group = series.create_group(self.param_name)
        self.param_frequency = 2
        self.param_supf_offset = 1.5
        self.param_arinc_429 = True
        param_group.attrs['frequency'] = self.param_frequency
        param_group.attrs['supf_offset'] = self.param_supf_offset
        param_group.attrs['arinc_429'] = self.param_arinc_429
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
        hdf.close()
        self.hdf_file = hdf_file(self.hdf_path)

    def tearDown(self):
        if self.hdf_file.hdf.id:
            self.hdf_file.close()
        os.remove(self.hdf_path)

    def test_dependency_tree(self):
        self.assertEqual(self.hdf_file.dependency_tree, None)
        dependency_tree = {'Airspeed': ['Altitude AAL'], 'Altitude AAL': []}
        self.hdf_file.dependency_tree = dependency_tree
        self.assertEqual(self.hdf_file.dependency_tree, dependency_tree)
        self.hdf_file.dependency_tree = None
        self.assertEqual(self.hdf_file.dependency_tree, None)

    def test_duration(self):
        self.assertEqual(self.hdf_file.duration, None)
        self.hdf_file.duration = 1.5
        self.assertEqual(self.hdf_file.duration, 1.5)
        self.hdf_file.duration = None
        self.assertEqual(self.hdf_file.duration, None)

    def test_start_datetime(self):
        self.assertEqual(self.hdf_file.start_datetime, None)
        datetime_1 = datetime.now()
        timestamp = calendar.timegm(datetime_1.utctimetuple())
        self.hdf_file.start_datetime = timestamp
        # Microsecond accuracy is lost.
        self.assertEqual(self.hdf_file.start_datetime,
                         datetime_1.replace(microsecond=0))
        datetime_2 = datetime.now()
        self.hdf_file.start_datetime = datetime_2
        self.assertEqual(self.hdf_file.start_datetime,
                         datetime_2.replace(microsecond=0))
        self.hdf_file.start_datetime = None
        self.assertEqual(self.hdf_file.start_datetime, None)

    def test_version(self):
        self.assertEqual(self.hdf_file.version, None)
        self.hdf_file.version = '0.1.2'
        self.assertEqual(self.hdf_file.version, '0.1.2')
        self.hdf_file.version = None
        self.assertEqual(self.hdf_file.version, None)

    def test_get_matching(self):
        regex_str = '^TEST_PARAM10$'
        params = self.hdf_file.get_matching(regex_str)
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].name, self.param_name)
        regex_str = '^TEST_PARAM1[01]$'
        params = self.hdf_file.get_matching(regex_str)
        self.assertEqual(len(params), 2)
        param_names = [p.name for p in params]
        self.assertTrue(self.param_name in param_names)
        self.assertTrue(self.masked_param_name in param_names)

    def test_open_and_close_and_full_masks(self):
        self.hdf_file.close()
        with hdf_file(self.hdf_path) as hdf:
            # check it's open
            self.assertFalse(hdf.hdf.id is None)
            hdf['sample'] = Parameter('sample', np.array(range(10)))
            self.assertEqual(list(hdf['sample'].array.data), range(10))
            self.assertTrue(hasattr(hdf['sample'].array, 'mask'))

            hdf['masked sample'] = Parameter('masked sample', np.ma.array(range(10)))
            self.assertEqual(list(hdf['masked sample'].array.data), range(10))
            # check masks are returned in full (not just a single False)
            self.assertEqual(list(hdf['masked sample'].array.mask), [False] * 10)
        # check it's closed
        self.assertEqual(hdf.hdf.__repr__(), '<Closed HDF5 file>')
        #self.assertEqual(hdf.duration, None) # Cannot access closed file attribute.

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

        self.hdf_file.set_param_limits(self.param_name,
                                       parameter_limits.get('TEST_PARAM10'))

        stored_param_limits = self.hdf_file.get_param_limits(self.param_name)
        self.assertEqual(parameter_limits.get('TEST_PARAM10'),
                         stored_param_limits)

    def test_get_params(self):
        hdf_file = self.hdf_file
        # Test retrieving all parameters.
        params = hdf_file.get_params()
        self.assertTrue(len(params) == 2)
        param = params['TEST_PARAM10']
        self.assertEqual(param.frequency, self.param_frequency)
        self.assertEqual(param.arinc_429, self.param_arinc_429)
        param = params['TEST_PARAM11']
        self.assertEqual(param.offset, self.masked_param_supf_offset)
        self.assertEqual(param.arinc_429, None)
        # Test retrieving single specified parameter.
        params = hdf_file.get_params(param_names=['TEST_PARAM10'])
        self.assertTrue(len(params) == 1)
        param = params['TEST_PARAM10']
        self.assertEqual(param.frequency, self.param_frequency)
  
    def test_get_param_valid_only(self):
        hdf = self.hdf_file
        hdf['Valid Param'] = Parameter('Valid Param', array=np.ma.arange(10))
        hdf['Invalid Param'] = Parameter('Invalid Param', array=np.ma.arange(10), invalid=True)
        self.assertIn('Invalid Param', hdf)
        # request Invalid param without filtering
        self.assertTrue(hdf.get_param('Invalid Param', valid_only=False))
        # filtering only valid raises keyerror
        self.assertRaises(KeyError, hdf.get_param, 'Invalid Param', valid_only=True)
                
        
    def test_get_params_valid_only(self):
        hdf = self.hdf_file
        hdf['Valid Param'] = Parameter('Valid Param', array=np.ma.arange(10))
        hdf['Invalid Param'] = Parameter('Invalid Param', array=np.ma.arange(10), invalid=True)
        self.assertIn('Invalid Param', hdf)
        # check the params that are valid are listed correctly
        self.assertEqual(hdf.valid_param_names(), 
                         ['TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
        # request all params inc. invalid
        all_params = hdf.get_params(valid_only=False)
        self.assertEqual(sorted(all_params.keys()),
                         ['Invalid Param', 'TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
        # request only valid params
        valid = hdf.get_params(valid_only=True)
        self.assertEqual(sorted(valid.keys()), 
                         ['TEST_PARAM10', 'TEST_PARAM11', 'Valid Param'])
        # request params by name
        valid_few = hdf.get_params(param_names=['Valid Param'], valid_only=True)
        self.assertEqual(valid_few.keys(), 
                         ['Valid Param'])
        # try to request the invalid param, but only accepting valid raises keyerror
        self.assertRaises(KeyError, hdf.get_params, 
                          param_names=['Invalid Param', 'Valid Param'],
                          valid_only=True, raise_keyerror=True)
        
                         

    def test___set_item__(self):
        '''
        set_item uses set_param
        '''
        set_param_data = self.hdf_file.__setitem__
        hdf_file = self.hdf_file
        series = hdf_file.hdf['series']
        # Create new parameter with np.array.
        name1 = 'TEST_PARAM1'
        array = np.arange(100)
        set_param_data(name1, Parameter(name1, array))
        self.assertTrue(np.all(series[name1]['data'].value == array))
        self.assertFalse('arinc_429' in series[name1].attrs)
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
        self.assertTrue(np.all(series[name2]['data'].value == array))
        self.assertTrue(np.all(series[name2]['mask'].value == mask))
        self.assertEqual(series[name2].attrs['frequency'], param2_frequency)
        self.assertEqual(series[name2].attrs['supf_offset'], param2_offset)
        self.assertEqual(series[name2].attrs['arinc_429'], param2_arinc_429)

        # Set existing parameter's data with np.array.
        array = np.arange(200)
        set_param_data(name1, Parameter(name1, array))
        self.assertTrue(np.all(series[name1]['data'].value == array))

        # Set existing parameter's data with np.ma.masked_array.
        mask = [bool(random.randint(0, 1)) for x in range(len(array))]
        masked_array = np.ma.masked_array(data=array, mask=mask)
        set_param_data(name1, Parameter(name1, masked_array))
        self.assertTrue(np.all(series[name1]['data'].value == array))
        self.assertTrue(np.all(series[name1]['mask'].value == mask))

    def test_update_param_mask(self):
        # setup original array
        name1 = 'Airspeed Minus Vref'
        array = np.arange(200)
        self.hdf_file.set_param(Parameter(name1, array))
        series = self.hdf_file.hdf['series']

        # Set mask changes, but not Data array
        new_array = np.arange(200)[::-1]
        new_mask = [bool(random.randint(0, 1)) for x in range(len(new_array))]
        masked_array = np.ma.masked_array(data=new_array, mask=new_mask)

        self.hdf_file.set_param(Parameter(name1, masked_array),
                                save_data=False, save_mask=True)
        # assert new mask is saved
        self.assertTrue(np.all(series[name1]['mask'].value == new_mask))
        # asssert data remains same as original array
        self.assertTrue(np.all(series[name1]['data'].value == array))

        # This test is not currently performed:
        ### check is conducted to ensure data and mask are still the same length
        ### (and that the data exists)
        ##shorter_masked_array = np.ma.arange(100)
        ##self.assertRaises(np.ma.core.MaskError, hdf_file.set_param,
                          ##Parameter(name1, shorter_masked_array),
                          ##dict(save_data=False))

    def test_update_param_attributes(self):
        # save initial Parameter to file
        self.hdf_file.set_param(Parameter('Blah', np.ma.arange(1)))
        #Update the invalidity flag only
        self.hdf_file.set_param(Parameter('Blah', np.ma.arange(1), invalid=1),
                                save_data=False, save_mask=False)
        self.assertEqual(self.hdf_file['Blah'].invalid, 1)

    def test_get_param_data(self):
        self.__test_get_param_data(self.hdf_file.get_param)

    def test___get_item__(self):
        self.__test_get_param_data(self.hdf_file.__getitem__)

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

    def test_len(self):
        '''
        Depends upon HDF creation in self.setUp().
        '''
        self.assertEqual(len(self.hdf_file), 2)

    def test_keys(self):
        '''
        Depends upon HDF creation in self.setUp().
        '''
        self.assertEqual(sorted(self.hdf_file.keys()),
                         sorted([self.param_name, self.masked_param_name]))

    def test_startswith(self):
        params = ('Airspeed Two', 'Airspeed One', 'blah')
        mock_keys = mock.Mock(spec=['keys'], return_value=params)
        self.hdf_file.keys = mock_keys

        self.assertEqual(self.hdf_file.startswith('Airspeed'),
                         ['Airspeed One', 'Airspeed Two'])

    def test_search(self):
        """
        """
        params = ['ILS Localizer', 'ILS Localizer (R)', 'ILS Localizer (L)', 'Rate of Climb', 'Altitude STD',
                  'Brake (R) Pressure Ourboard', 'Brake (L) Pressure Inboard', 'ILS Localizer Deviation Warning',
                  'ILS Localizer Test Tube Inhibit', 'ILS Localizer Beam Anomaly', 'ILS Localizer Engaged']

        mock_keys = mock.Mock(spec=['keys'], return_value=params)
        self.hdf_file.keys = mock_keys

        search_key = 'ILS Localizer'

        expected_output = ['ILS Localizer', 'ILS Localizer (L)', 'ILS Localizer (R)', 'ILS Localizer Beam Anomaly', 'ILS Localizer Deviation Warning',
                  'ILS Localizer Engaged', 'ILS Localizer Test Tube Inhibit']

        res = self.hdf_file.search(search_key)
        self.assertEqual(res, expected_output)

        search_key_star = 'ILS Localizer (*)'

        expected_output_star = ['ILS Localizer (L)', 'ILS Localizer (R)']
        res = self.hdf_file.search(search_key_star)
        self.assertEqual(res, expected_output_star)

    def test_mapped_array(self):
        # created mapped array
        mapping = {0: 'zero', 2: 'two', 3: 'three'}
        array = np.ma.array(range(5) + range(5), mask=[1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        multi_p = Parameter('multi', array, values_mapping=mapping)
        multi_p.array[0] = 'three'

        # save array to hdf
        self.hdf_file['multi'] = multi_p
        self.hdf_file.close()

        # check hdf has mapping and integer values stored
        with hdf_file(self.hdf_path) as hdf:
            saved = hdf['multi']
            self.assertEqual(str(saved.array[:]),
                             '[three -- -- three ? zero ? two -- --]')
            self.assertEqual(saved.array.data.dtype, np.int)

    def test__delitem__(self):
        p = 'TEST_PARAM10'
        self.assertTrue(p in self.hdf_file)
        del self.hdf_file[p]
        self.assertFalse(p in self.hdf_file)
        self.assertFalse(p in self.hdf_file.hdf['series'])

        self.assertRaises(KeyError, self.hdf_file.__delitem__, 'INVALID_PARAM_NAME')

    def test_delete_params(self):
        ps = ['TEST_PARAM10', 'TEST_PARAM11', 'INVALID_PARAM_NAME']
        self.assertTrue(ps[0] in self.hdf_file)
        self.assertTrue(ps[1] in self.hdf_file)
        self.assertFalse(ps[2] in self.hdf_file)  # invalid is not there
        # delete and ensure by default keyerrors are supressed
        self.hdf_file.delete_params(ps)
        for p in ps:
            self.assertFalse(p in self.hdf_file)

    def test_create_file(self):
        temp = 'temp_new_file.hdf5'
        if os.path.exists(temp):
            os.remove(temp)
        # cannot create file without specifying 'create=True'
        self.assertRaises(IOError, hdf_file, temp)
        self.assertFalse(os.path.exists(temp))
        # this one will create the file
        hdf = hdf_file(temp, create=True)
        self.assertTrue(os.path.exists(temp))
        self.assertEqual(hdf.hdfaccess_version, 1)
        os.remove(temp)

    def test_set_and_get_attributes(self):
        # Test setting a datetime as it's a non-json non-string type.
        self.assertFalse(self.hdf_file.hdf.attrs.get('start_datetime'))
        self.hdf_file.set_attr('start_datetime', datetime.now())
        self.assertEqual(self.hdf_file.get_attr('non-existing'), None)
        self.assertEqual(self.hdf_file.get_attr('non-existing',
                                                default='default'), 'default')
        # ensure that HDF is still working after keyerror raised!
        self.assertTrue('start_datetime' in self.hdf_file.hdf.attrs)
