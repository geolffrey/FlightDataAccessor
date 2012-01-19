import os
import re
import numpy as np
import h5py
try:
    import json
except ImportError:
    import simplejson as json

from utilities.filesystem_tools import pretty_size

from hdfaccess.parameter import Parameter


class hdf_file(object):    # rare case of lower case?!
    """ usage example:
    with hdf_file('path/to/file.hdf5') as hdf:
        print hdf['Altitude AAL']['data'][:20]
        
    # bits of interest
    hdf['series']['Altitude AAL']['levels']['1'] (Float array)
    hdf['series']['Altitude AAL']['data'] (Float array)
    hdf['series']['Altitude AAL']['mask'] (Bool list)
    hdf['series']['Altitude AAL'].attrs['limits'] (json)
    """
    # HDF file settings should be consistent, therefore hardcoding defaults.
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 3}
    
    def __repr__(self):
        '''
        Q: What else should be displayed?
        '''
        size = pretty_size(os.path.getsize(self.hdf.filename))
        return '%s %s (%d parameters)' % (self.hdf.filename, size, len(self))
    
    def __str__(self):
        return self.__repr__()
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.hdf = h5py.File(self.file_path, 'r+')
        self.duration = self.hdf.attrs.get('duration')
        rfc = self.hdf.attrs.get('reliable_frame_counter', 0)
        self.reliable_frame_counter = rfc == 1
                
    def __enter__(self):
        '''
        HDF file is opened on __init__.
        '''
        return self

    def __exit__(self, a_type, value, traceback):
        self.close()
        
    def __getitem__(self, key):
        '''
        Allows for dictionary access: hdf['Altitude AAL'][:30]
        '''
        return self.get_param(key)
        
    def __setitem__(self, key, param):
        """ Allows for: hdf['Altitude AAL'] = np.ma.array()
        """
        assert key == param.name
        return self.set_param(param)
    
    def __contains__(self, key):
        """check if the key exists"""
        return key in self.keys()
    
    def __len__(self):
        '''
        Number of parameter groups within the series group.
        
        :returns: Number of parameters.
        :rtype: int
        '''
        return len(self.hdf['series'])
    
    def keys(self):
        '''
        Parameter group names within the series group.
        
        :returns: List of parameter names.
        :rtype: list of str
        '''
        return self.hdf['series'].keys()
    get_param_list = keys
    
    def close(self):
        self.hdf.flush() # Q: required?
        self.hdf.close()
        
    def search(self, term):
        '''
        Searches for partial matches of term within keys.
        '''
        return sorted(filter(lambda x: term.upper() in x.upper(), self.keys()))
        
    
    def get_params(self, param_names=None):
        '''
        Returns params that are available, `ignores` those that aren't.
    
        :param param_names: Parameters to return, if None returns all parameters
        :type param_names: list of str or None
        :returns: Param name to Param object dict
        :rtype: dict
        '''
        if param_names is None:
            param_names = self.keys()
        param_name_to_obj = {}
        for name in param_names:
            try:
                param_name_to_obj[name] = self[name]
            except KeyError:
                pass # ignore parameters that aren't available
        return param_name_to_obj

    def get_param(self, name):
        '''
        name e.g. "Head Mag"
        Returns a masked_array. If 'mask' is stored it will be the mask of the
        returned masked_array, otherwise it will be False.
        
        :param name: Name of parameter with 'series'.
        :type name: str
        :returns: Parameter object containing HDF data and attrs.
        :rtype: Parameter
        '''
        if name not in self:
            # catch exception otherwise HDF will crash and close
            raise KeyError("%s" % name)
        param_group = self.hdf['series'][name]
        data = param_group['data']
        mask = param_group.get('mask', False)
        array = np.ma.masked_array(data, mask=mask)
        kwargs = {}
        if 'frequency' in param_group.attrs:
            kwargs['frequency'] = param_group.attrs['frequency']
        if 'latency' in param_group.attrs:
            # Differing terms: latency is known internally as frame offset.
            kwargs['offset'] = param_group.attrs['latency']
        if 'arinc_429' in param_group.attrs:
            kwargs['arinc_429'] = param_group.attrs['arinc_429']
        return Parameter(name, array, **kwargs)
    
    def get(self, name, default=None):
        """
        Dictionary like .get operator
        """
        try:
            return self.get_param(name)
        except KeyError:
            return default
    
    def get_or_create(self, param_name):
        # Either get or create parameter.
        if param_name in self.hdf['series']:
            param_group = self.hdf['series'][param_name]
        else:
            param_group = self.hdf['series'].create_group(param_name)
            param_group.attrs['name'] = param_name
            param_group.attrs['external_datatype'] = 'float'
            param_group.attrs['external_dataformat'] = '%.2f'
        return param_group

    def set_param(self, param):
        '''
        Store parameter and associated attributes on the HDF file.
        
        Parameter.name canot contain forward slashes as they are used as an
        HDF identifier which supports filesystem-style indexing, e.g.
        '/series/CAS'.
        
        :param param: Parameter like object with attributes name (must not contain forward slashes), array. 
        :param array: Array containing data and potentially a mask for the data.
        :type array: np.array or np.ma.masked_array
        '''
        # Allow both arrays and masked_arrays.
        if hasattr(param.array, 'mask'):
            array = param.array
        else:
            array = np.ma.masked_array(param.array, mask=False)
            
        param_group = self.get_or_create(param.name)
        if 'data' in param_group:
             # Dataset must be deleted before recreation.
            del param_group['data']
        dataset = param_group.create_dataset('data', data=array.data, 
                                             **self.DATASET_KWARGS)
        if 'mask' in param_group:
            # Existing mask will no longer reflect the new data.
            del param_group['mask']
        mask = np.ma.getmaskarray(array)
        mask_dataset = param_group.create_dataset('mask', data=mask,
                                                  **self.DATASET_KWARGS)
        # Set parameter attributes
        param_group.attrs['latency'] = param.offset
        param_group.attrs['frequency'] = param.frequency
        if hasattr(param, 'arinc_429') and param.arinc_429 is not None:
            # A None value cannot be stored within the HDF file as an attribute.
            param_group.attrs['arinc_429'] = param.arinc_429
        #TODO: param_group.attrs['available_dependencies'] = param.available_dependencies
        #TODO: Possible to store validity percentage upon name.attrs
    
    def set_param_limits(self, name, limits):
        '''
        Stores limits for a parameter in JSON format.
        
        :param name: Parameter name
        :type name: str
        :param limits: Operating limits storage
        :type limits: dict
        '''
        param_group = self.get_or_create(name)
        param_group.attrs['limits'] = json.dumps(limits)
        
    def get_param_limits(self, name):
        '''
        Returns a parameter's operating limits stored within the groups
        'limits' attribute. Decodes limits from JSON into dict.
        
        :param name: Parameter name
        :type name: str
        :returns: Parameter operating limits or None if 'limits' attribute does not exist.
        :rtype: dict or None
        :raises KeyError: If parameter name does not exist within the HDF file.
        '''
        if name not in self:
            # Do not try to retrieve a non-existing group within the HDF 
            # otherwise h5py.File object will crash and close.
            raise KeyError("%s" % name)
        limits = self.hdf['series'][name].attrs.get('limits')
        return json.loads(limits) if limits else None
    
    def get_matching(self, regex_str):
        '''
        Get parameters with names matching regex_str.
        
        :param regex_str: Regex to match against parameters.
        :type regex_str: str
        :returns: Parameters which match regex_str.
        :rtype: list of Parameter
        '''
        compiled_regex = re.compile(regex_str)
        param_names = filter(compiled_regex.match, self.keys())
        return [self[param_name] for param_name in param_names]


def print_hdf_info(hdf_file):
    hdf_file = hdf_file.hdf
    series = hdf_file['series']
    # IOLA
    # 8.0
    if 'Time' in series:
        print 'Tailmark:', hdf_file.attrs['tailmark']
        print 'Start Time:', hdf_file.attrs['starttime']
        print 'End Time:', hdf_file.attrs['endtime']
    
    for group_name, group in series.iteritems():
        print '[%s]' % group_name
        print 'Frequency:', group.attrs['frequency']
        print group.attrs.items()
        # IOLA's latency is our frame offset.
        print 'Offset:', group.attrs['latency']
        print 'External Data Type:', group.attrs['external_datatype']
        print 'External Data Format:', group.attrs['external_dataformat']
        print 'Number of recorded values:', len(group['data'])
    #param_series = hdf_file['series'][parameter]
    #data = param_series['data']


if __name__ == '__main__':
    import sys
    print_hdf_info(hdf_file(sys.argv[1]))
    sys.exit()
    file_path = 'AnalysisEngine/resources/data/hdf5/flight_1626325.hdf5'    
    with hdf_file(file_path) as hdf:
        print_hdf_info(hdf)
        
    hdf = h5py.File(
        'AnalysisEngine/resources/data/hdf5/flight_1626326.hdf5', 'w')
    hdf['series']['Altitude AAL'].attrs['limits'] = {'min':0,  'max':50000}
    