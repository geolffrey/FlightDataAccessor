import h5py
import numpy as np
import os
import random
import unittest

from hdfaccess.file import hdf_file
from hdfaccess.parameter import Parameter

TEST_DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
TEMP_DIR_PATH = os.path.join(TEST_DATA_DIR_PATH, 'temp')
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'test_data')

class TestHdfFile(unittest.TestCase):
    def setUp(self):
        self.hdf_path = os.path.join(TEST_DATA_DIR, 'test_hdf_access.hdf')
        hdf = h5py.File(self.hdf_path, 'w')
        series = hdf.create_group('series')
        self.param_name = 'TEST_PARAM10'
        param_group = series.create_group(self.param_name)
        self.param_frequency = 2
        self.param_latency = 1.5
        self.param_arinc_429 = True
        param_group.attrs['frequency'] = self.param_frequency
        param_group.attrs['latency'] = self.param_latency
        param_group.attrs['arinc_429'] = self.param_arinc_429
        self.param_data = np.arange(100)
        dataset = param_group.create_dataset('data', data=self.param_data)
        self.masked_param_name = 'TEST_PARAM11'
        masked_param_group = series.create_group(self.masked_param_name)
        self.masked_param_frequency = 4
        self.masked_param_latency = 2.5
        masked_param_group.attrs['frequency'] = self.masked_param_frequency
        masked_param_group.attrs['latency'] = self.masked_param_latency
        self.param_mask = [bool(random.randint(0, 1)) for x in range(len(self.param_data))]
        dataset = masked_param_group.create_dataset('data', data=self.param_data)
        mask_dataset = masked_param_group.create_dataset('mask', data=self.param_mask)
        hdf.close()
        self.hdf_file = hdf_file(self.hdf_path)
    
    def tearDown(self):
        if self.hdf_file.hdf.id:
            self.hdf_file.close()
        os.remove(self.hdf_path)
        
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
            self.assertEqual(list(hdf['masked sample'].array.mask), [False]*10)
        # check it's closed
        self.assertEqual(hdf.hdf.__repr__(), '<Closed HDF5 file>')
        self.assertEqual(hdf.duration, None)
        
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
        self.assertEqual(param.offset, self.masked_param_latency)
        self.assertEqual(param.arinc_429, None)
        # Test retrieving single specified parameter.
        params = hdf_file.get_params(param_names=['TEST_PARAM10'])
        self.assertTrue(len(params) == 1)
        param = params['TEST_PARAM10']
        self.assertEqual(param.frequency, self.param_frequency)
        
            
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
        self.assertEqual(series[name2].attrs['latency'], param2_offset)
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
        self.assertEqual(self.param_latency, param.offset)
        # Create new parameter with np.array.
        param = get_param(self.masked_param_name)
        self.assertTrue(np.all(self.param_data == param.array.data))
        self.assertTrue(np.all(self.param_mask == param.array.mask))
        self.assertEqual(self.masked_param_frequency, param.frequency)
        self.assertEqual(self.masked_param_latency, param.offset)
    
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
        self.hdf_file.keys.return_value = ('Airspeed Two', 'Airspeed One', 'blah')
        self.assertEqual(self.hdf_file.startswith('Airspeed'),
                         ('Airspeed One', 'Airspeed Two'))
