import base64
import calendar
import logging
import h5py
import numpy as np
import os
import pickle
import re
import simplejson
import zlib

from copy import deepcopy
from datetime import datetime

from flightdatautilities.filesystem_tools import pretty_size
from flightdatautilities.patterns import wildcard_match

from hdfaccess.parameter import Parameter


HDFACCESS_VERSION = 1


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
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 6}

    def __repr__(self):
        '''
        Q: What else should be displayed?
        '''
        size = pretty_size(os.path.getsize(self.hdf.filename))
        return '%s %s (%d parameters)' % (self.hdf.filename, size, len(self))

    def __str__(self):
        return self.__repr__()

    def __init__(self, file_path_or_obj, cache_param_list=[], create=False):
        '''
        Opens an HDF file (or accepts and already open h5py.File object) - will
        create if does not exist if create=True!

        :param cache_param_list: Names of parameters to cache where accessed
        :type cache_param_list: list of str
        :param file_path_or_obj: Can be either the path to an HDF file or an already opened HDF file object.
        :type file_path_or_obj: str or h5py.File
        :param create: ill allow creation of file if it does not exist.
        :type create: bool
        '''
        if isinstance(file_path_or_obj, h5py.File):
            hdf_exists = True
            self.hdf = file_path_or_obj
            if self.hdf.mode != 'r+':
                raise ValueError("hdf_file requires mode 'r+'.")
            self.file_path = os.path.abspath(self.hdf.filename)
        else:
            hdf_exists = os.path.isfile(file_path_or_obj)
            if not create and not hdf_exists:
                raise IOError('File not found: %s' % file_path_or_obj)
            self.file_path = os.path.abspath(file_path_or_obj)
            # Not specifying a mode, will create the file if the path does not
            # exist and open with mode 'r+'.
            self.hdf = h5py.File(self.file_path)

        self.hdfaccess_version = self.hdf.attrs.get('hdfaccess_version', 1)
        if hdf_exists:
            # default version is 1
            assert self.hdfaccess_version == HDFACCESS_VERSION
        else:
            # just created this file, add the current version
            self.hdf.attrs['hdfaccess_version'] = HDFACCESS_VERSION

        rfc = self.hdf.attrs.get('reliable_frame_counter', 0)
        self.reliable_frame_counter = rfc == 1

        if 'series' not in self.hdf.keys():
            # The 'series' group is required for storing parameters.
            self.hdf.create_group('series')
        # cache keys as accessing __iter__ on hdf groups is v.slow
        self._keys_cache = None
        self._valid_param_names_cache = None
        # cache parameters that are used often
        self._params_cache = {}
        # this is the list of parameters to cache
        self.cache_param_list = cache_param_list

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

    def iteritems(self):
        """
        """
        for param_name in self.keys():
            yield param_name, self[param_name]

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
        if not self._keys_cache:
            series = self.hdf['series']
            keys = series.keys()
            self._keys_cache = sorted(keys)
        return self._keys_cache
    get_param_list = keys

    def lfl_keys(self):
        '''
        Parameter group names within the series group which came from the
        Logical Frame Layout.

        :returns: List of LFL parameter names.
        :rtype: list of str
        '''
        lfl_keys = []
        for param_name in self.keys():
            if self.hdf['series'][param_name].attrs.get('lfl'):
                lfl_keys.append(param_name)
        return lfl_keys

    def derived_keys(self):
        '''
        Parameter group names within the series group which are derived
        parameters.

        :returns: List of derived parameter names.
        :rtype: list of str
        '''
        derived_keys = []
        for param_name in self.keys():
            if not self.hdf['series'][param_name].attrs.get('lfl'):
                derived_keys.append(param_name)
        return derived_keys

    def close(self):
        self.hdf.flush()  # Q: required?
        self.hdf.close()

    # HDF Attribute properties
    ############################################################################

    @property
    def analysis_version(self):
        '''
        Accessor for the root-level 'version' attribute.

        :returns: Version of the FlightDataAnalyzer which processed this HDF file.
        :rtype: str or None
        '''
        return self.hdf.attrs.get('analysis_version')

    @analysis_version.setter
    def analysis_version(self, analysis_version):
        '''
        Mutator for the root-level 'version' attribute. If version is None the
        'version' attribute will be deleted.

        :param version: FlightDataAnalyser version.
        :type version: str
        :rtype: None
        '''
        if analysis_version is None:  # Cannot store None as an HDF attribute.
            if 'analysis_version' in self.hdf.attrs:
                del self.hdf.attrs['analysis_version']
        else:
            self.hdf.attrs['analysis_version'] = analysis_version

    @property
    def dependency_tree(self):
        '''
        Accessor for the root-level 'dependency_tree' attribute.

        :rtype: list or None
        '''
        dependency_tree = self.hdf.attrs.get('dependency_tree')
        if dependency_tree:
            return simplejson.loads(
                zlib.decompress(base64.decodestring(dependency_tree)))
        else:
            return None

    @dependency_tree.setter
    def dependency_tree(self, dependency_tree):
        '''
        Mutator for the root-level 'dependency_tree' attribute. If
        dependency_tree is None the 'dependency_tree' attribute will be deleted.
        The attribute is bz2 compressed due to the 64KB attribute size
        limit of the HDF file and encoded with base64 to avoid 'ValueError:
        VLEN strings do not support embedded NULLs' when compressed data
        includes null characters.

        :param dependency_tree: Dependency tree created by the FlightDataAnalyser during processing.
        :rtype: None
        '''
        if dependency_tree is None:
            if 'dependency_tree' in self.hdf.attrs:
                del self.hdf.attrs['dependency_tree']
        else:
            self.hdf.attrs['dependency_tree'] = \
                base64.encodestring(
                    zlib.compress(simplejson.dumps(dependency_tree)))

    @property
    def duration(self):
        '''
        Accessor for the root-level 'duration' attribute.

        :rtype: float or None
        '''
        duration = self.hdf.attrs.get('duration')
        return float(duration) if duration else None

    @duration.setter
    def duration(self, duration):
        '''
        Mutator for the root-level 'duration' attribute. If duration is None the
        'duration' attribute will be deleted.

        :param duration: Duration of this file's data in seconds.
        :type duration: float
        :rtype: None
        '''
        if duration is None:  # Cannot store None as an HDF attribute.
            if 'duration' in self.hdf.attrs:
                del self.hdf.attrs['duration']
        else:
            self.hdf.attrs['duration'] = duration

    @property
    def reliable_frame_counter(self):
        '''
        Accessor for the root-level 'reliable_frame_counter' attribute.

        :rtype: bool or None
        '''
        reliable_frame_counter = self.hdf.attrs.get('reliable_frame_counter')
        return bool(reliable_frame_counter) if reliable_frame_counter is not None else None

    @reliable_frame_counter.setter
    def reliable_frame_counter(self, reliable_frame_counter):
        '''
        Mutator for the root-level 'reliable_frame_counter' attribute.
        If reliable_frame_counter is None the 'reliable_frame_counter' attribute
        will be deleted.

        :param reliable_frame_counter: Flag indicating whether frame counter is reliable
        :type reliable_frame_counter: bool
        :rtype: None
        '''
        if reliable_frame_counter is None:  # Cannot store None as an HDF attribute.
            if 'reliable_frame_counter' in self.hdf.attrs:
                del self.hdf.attrs['reliable_frame_counter']
        else:
            self.hdf.attrs['reliable_frame_counter'] = 1 if reliable_frame_counter else 0

    @property
    def start_datetime(self):
        '''
        The start datetime of the data stored within the HDF file.

        Converts the root-level 'start_timestamp' attribute from a timestamp to
        a datetime.

        :returns: Start datetime if 'start_timestamp' is set, otherwise None.
        :rtype: datetime or None
        '''
        timestamp = self.hdf.attrs.get('start_timestamp')
        return datetime.utcfromtimestamp(timestamp) if timestamp else None

    @start_datetime.setter
    def start_datetime(self, start_datetime):
        '''
        Converts start_datetime to a timestamp and saves as 'start_timestamp'
        root-level attribute. If start_datetime is None the 'start_timestamp'
        attribute will be deleted.

        :param start_datetime: The datetime at the beginning of this file's data.
        :type start_datetime: datetime or timestamp
        :rtype: None
        '''
        if start_datetime is None:
            if 'start_timestamp' in self.hdf.attrs:
                del self.hdf.attrs['start_timestamp']
        else:
            if isinstance(start_datetime, datetime):
                timestamp = calendar.timegm(start_datetime.utctimetuple())
            else:
                timestamp = start_datetime
            self.hdf.attrs['start_timestamp'] = timestamp

    @property
    def superframe_present(self):
        '''
        Whether or the frame which was used to create the HDF file had a
        superframe counter.

        Accessor for the root-level 'superframe_present' attribute.

        :rtype: bool or None
        '''
        superframe_present = self.hdf.attrs.get('superframe_present')
        return bool(superframe_present) if superframe_present is not None else None

    @superframe_present.setter
    def superframe_present(self, superframe_present):
        '''
        Mutator for the root-level 'superframe_present' attribute. If superframe_present is None the
        'superframe_present' attribute will be deleted.

        :param superframe_present: Flag indicating whether superframes are recorded
        :type superframe_present: bool
        :rtype: None
        '''
        if superframe_present is None:  # Cannot store None as an HDF attribute.
            if 'superframe_present' in self.hdf.attrs:
                del self.hdf.attrs['superframe_present']
        else:
            self.hdf.attrs['superframe_present'] = 1 if superframe_present else 0

    @property
    def version(self):
        '''
        Accessor for the root-level 'version' attribute.

        :returns: The version of downsampling applied to the HDF file.
        :rtype: str or None
        '''
        return self.hdf.attrs.get('version')

    @version.setter
    def version(self, version):
        '''
        Mutator for the root-level 'version' attribute. If version is None the
        'version' attribute will be deleted.

        :param version: The version of downsampling applied to the HDF file.
        :type version: str
        :rtype: None
        '''
        if version is None:  # Cannot store None as an HDF attribute.
            if 'version' in self.hdf.attrs:
                del self.hdf.attrs['version']
        else:
            self.hdf.attrs['version'] = version

    def get_attr(self, name, default=None):
        '''
        Get an attribute stored on the hdf.

        :param name: Key name for attribute to be recalled.
        :type name: String
        '''
        value = self.hdf.attrs.get(name)
        if value:
            return pickle.loads(value)
        else:
            return default

    def set_attr(self, name, value):
        '''
        Store an attribute on the hdf at the top level. Objects are pickled to
        ASCII.

        Note: 64KB might get used up quite quickly!

        :param name: Key name for attribute to be stored
        :type name: String
        :param value: Value to store as an attribute
        :type value: any
        '''
        if value is None:  # Cannot store None as an HDF attribute.
            if name in self.hdf.attrs:
                del self.hdf.attrs[name]
            return
        else:
            self.hdf.attrs[name] = pickle.dumps(value, protocol=0)
            return

    # HDF Accessors
    ##########################################################################

    def search(self, pattern, lfl_keys_only=False):
        '''
        Searches for param names that matches with (*) or (?) expression. If
        found, the pattern is converted to a regex and matched against the
        param names in the hdf file. If a match is found, the param is added
        as a key in a list and returned.

        If a match with the regular expression is not found, then a list of
        params are returned that contains the substring 'pattern'.

        :param pattern: Pattern to search for (case insensitve).
        :type pattern: string
        :param lfl_keys_only: Search only within LFL keys
        :type lfl_keys_only: boolean
        :returns: list of sorted keys(params)
        :rtype: list
        '''
        if lfl_keys_only:
            keys = self.lfl_keys()
        else:
            keys = self.keys()
        if '(*)' in pattern or '(?)' in pattern:
            return wildcard_match(pattern, keys)
        else:
            PATTERN = pattern.upper()
            return sorted(
                filter(lambda k: PATTERN in k.upper(), keys))

    def startswith(self, term):
        '''
        Searches for keys which start with the term. Case sensitive.
        '''
        return sorted(filter(lambda x: x.startswith(term), self.keys()))

    def get_params(self, param_names=None, valid_only=False,
                   raise_keyerror=False, _slice=None):
        '''
        Returns params that are available, `ignores` those that aren't.

        :param param_names: Parameters to return, if None returns all parameters or all valid parameters if valid_only=True
        :type param_names: list of str or None
        :param valid_only: Only include valid parameters, by default invalid parameters are included.
        :type valid_only: Bool
        :param raise_keyerror: Whether to raise exception if parameter not in keys() or in valid keys if valid_only=True.
        :type raise_keyerror: Bool
        :param _slice: Only read a slice of the parameters' data. The slice indices are 1Hz.
        :type _slice: slice
        :returns: Param name to Param object dict
        :rtype: dict
        '''
        if param_names is None:
            if valid_only:
                param_names = self.valid_param_names()
            else:
                param_names = self.keys()
        param_name_to_obj = {}
        for name in param_names:
            try:
                param_name_to_obj[name] = self.get_param(
                    name, valid_only=valid_only, _slice=_slice)
            except KeyError:
                if raise_keyerror:
                    raise
                else:
                    pass  # ignore parameters that aren't available
        return param_name_to_obj

    def get_param(self, name, valid_only=False, _slice=None):
        '''
        name e.g. "Heading"
        Returns a masked_array. If 'mask' is stored it will be the mask of the
        returned masked_array, otherwise it will be False.

        :param name: Name of parameter with 'series'.
        :type name: str
        :param valid_only: Only return valid parameters, default is to include invalid params
        :type valid_only: bool
        :param _slice: Only read a slice of the parameter's data. The slice indices are 1Hz.
        :type _slice: slice
        :returns: Parameter object containing HDF data and attrs.
        :rtype: Parameter
        '''
        if valid_only and name not in self.valid_param_names():
            raise KeyError("%s" % name)
        elif name not in self:
            # catch exception otherwise HDF will crash and close
            raise KeyError("%s" % name)
        elif name in self._params_cache:
            logging.debug("Retrieving param '%s' from HDF cache", name)
            return deepcopy(self._params_cache[name])
        param_group = self.hdf['series'][name]
        data = param_group['data']
        mask = param_group.get('mask', False)  # FIXME: Replace False with a fully masked array

        kwargs = {}

        frequency = param_group.attrs.get('frequency', 1)
        kwargs['frequency'] = frequency
        
        if _slice and _slice.start is not None:
            slice_start = int(_slice.start * frequency)
        else:
            slice_start = 0
        if _slice and _slice.stop is not None:
            slice_stop = int(_slice.stop * frequency)
        else:
            slice_stop = len(data)
        
        data = data[slice_start:slice_stop]
        if mask:
            mask = mask[slice_start:slice_stop]
        
        # submasks
        # TODO: Read only the _slice of the submask from the file to speed up
        # segment splitting.
        kwargs['submasks'] = {}
        if 'submasks' in param_group.attrs and 'submasks' in param_group.keys():
            submask_map = simplejson.loads(param_group.attrs['submasks'])
            for submask_name, array_index in submask_map.items():
                kwargs['submasks'][submask_name] = \
                    param_group['submasks'][slice_start:slice_stop,array_index]

        array = np.ma.masked_array(data, mask=mask)

        if 'values_mapping' in param_group.attrs:
            mapping = simplejson.loads(param_group.attrs.get('values_mapping'))
            kwargs['values_mapping'] = mapping
        # Backwards compatibility. Q: When can this be removed?
        if 'supf_offset' in param_group.attrs:
            kwargs['offset'] = param_group.attrs['supf_offset']
        if 'arinc_429' in param_group.attrs:
            kwargs['arinc_429'] = param_group.attrs['arinc_429']
        if 'invalid' in param_group.attrs:
            kwargs['invalid'] = param_group.attrs['invalid']
            if kwargs['invalid'] and 'invalidity_reason' in param_group.attrs:
                kwargs['invalidity_reason'] = param_group.attrs['invalidity_reason']
        # Units
        if 'units' in param_group.attrs:
            kwargs['units'] = param_group.attrs['units']
        if 'lfl' in param_group.attrs:
            kwargs['lfl'] = param_group.attrs['lfl']
        elif 'description' in param_group.attrs:
            # Backwards compatibility for HDF files converted from AGS where
            # the units are stored in the description. Units will be invalid if
            # parameters from a FlightDataAnalyser HDF do not have 'units'
            # attributes.
            description = param_group.attrs['description']
            if description:
                kwargs['units'] = description
        if 'data_type' in param_group.attrs:
            kwargs['data_type'] = param_group.attrs['data_type']
        if 'source_name' in param_group.attrs:
            kwargs['source_name'] = param_group.attrs['source_name']
        if 'description' in param_group.attrs:
            kwargs['description'] = param_group.attrs['description']
        p = Parameter(name, array, **kwargs)
        # add to cache if required
        if name in self.cache_param_list:
            self._params_cache[name] = p
        return p

    def get(self, name, default=None):
        """
        Dictionary like .get operator.

        Makes no distinction on valid or invalid parameters that are requested.
        """
        try:
            return self.get_param(name)
        except KeyError:
            return default

    def get_or_create(self, param_name):
        '''
        Return a h5py parameter group, if it does not exist then create it too.
        '''
        # Either get or create parameter.
        if param_name in self:
            param_group = self.hdf['series'][param_name]
        else:
            self._keys_cache.append(param_name)  # Update cache.
            param_group = self.hdf['series'].create_group(param_name)
            param_group.attrs['name'] = str(param_name)  # Fails to set unicode attribute.
        return param_group

    def set_param(self, param, save_data=True, save_mask=True,
                  save_submasks=True):
        '''
        Store parameter and associated attributes on the HDF file.

        In order to save space re-writing data and masks to file when only
        one has changed or the attributes only have changed, the save_data
        and save_mask flags can be disabled as required.

        Parameter.name canot contain forward slashes as they are used as an
        HDF identifier which supports filesystem-style indexing, e.g.
        '/series/CAS'.

        :param param: Parameter like object with attributes name (must not
            contain forward slashes), array.
        :param array: Array containing data and potentially a mask for the
            data.
        :type array: np.array or np.ma.masked_array
        :param save_data: Whether or not to save the 'data' dataset.
        :type save_data: bool
        :param save_mask: Whether or not to save the 'mask' dataset.
        :type save_mask: bool
        :param save_submasks: Whether or not to save the 'submasks' dataset and 'submasks' attribute.
        :type save_submasks: bool
        '''
        if param.array.size == 0:
            raise ValueError('Data for parameter %s is empty! '
                             'Check the LFL (sample rate).' % param.name)

        # Allow both arrays and masked_arrays but ensure that we always have a fully expanded masked array.
        if not hasattr(param.array, 'mask'):  # FIXME: or param.mask == False:
            param.array = np.ma.masked_array(param.array, mask=False)

        if param.name in self.cache_param_list:
            logging.debug("Storing parameter '%s' in HDF cache", param.name)
            self._params_cache[param.name] = param  # FIXME is above?: Ensure that .mask is populated with np.ma.getmaskarray(param.array) to ensure we always have a full mask?

        param_group = self.get_or_create(param.name)
        if save_data:
            if 'data' in param_group:
                 # Dataset must be deleted before recreation.
                del param_group['data']
            param_group.create_dataset('data', data=param.array.data,
                                       **self.DATASET_KWARGS)
        if save_mask:
            if 'mask' in param_group:
                # Existing mask will no longer reflect the new data.
                del param_group['mask']
            mask = np.ma.getmaskarray(param.array)
            param_group.create_dataset('mask', data=mask,
                                       **self.DATASET_KWARGS)

        if save_submasks and hasattr(param, 'submasks') and param.submasks:
            if 'submasks' in param_group:
                del param_group['submasks']

            # Get array length for expanding booleans.
            submask_length = 0
            for submask_array in param.submasks.values():
                if (submask_array is None or
                    type(submask_array) in (bool, np.bool8)):
                    continue
                submask_length = max(submask_length, len(submask_array))

            submask_map = {}
            submask_arrays = []
            not_empty = (x for x in param.submasks.items() if x[1] is not None)
            for index, (submask_name, submask_array) in enumerate(not_empty):

                submask_map[submask_name] = index

                # Expand booleans to be arrays.
                if type(submask_array) in (bool, np.bool8):
                    function = np.ones if submask_array else np.zeros
                    submask_array = function(submask_length, dtype=np.bool8)
                submask_arrays.append(submask_array)

            param_group.create_dataset('submasks',
                                       data=np.column_stack(submask_arrays),
                                       **self.DATASET_KWARGS)
            param_group.attrs['submasks'] = simplejson.dumps(submask_map)

        # Set parameter attributes
        param_group.attrs['supf_offset'] = param.offset
        param_group.attrs['frequency'] = param.frequency
        # None values for arinc_429 and units cannot be stored within the
        # HDF file as an attribute.
        if hasattr(param, 'arinc_429') and param.arinc_429 is not None:
            param_group.attrs['arinc_429'] = param.arinc_429

        # Always store the validity state and reason(overwriting previous state)
        invalid = 1 if getattr(param, 'invalid', False) else 0
        param_group.attrs['invalid'] = invalid
        invalidity_reason = getattr(param, 'invalidity_reason', None) or ''
        param_group.attrs['invalidity_reason'] = invalidity_reason
        if hasattr(param, 'units') and param.units is not None:
            param_group.attrs['units'] = param.units
        if hasattr(param, 'lfl') and param.lfl is not None:
            param_group.attrs['lfl'] = param.lfl
        if hasattr(param, 'data_type') and param.data_type is not None:
            param_group.attrs['data_type'] = param.data_type
        if hasattr(param, 'values_mapping') and param.values_mapping:
            param_group.attrs['values_mapping'] = simplejson.dumps(
                param.values_mapping)
        if hasattr(param, 'source_name') and param.source_name:
            param_group.attrs['source_name'] = param.source_name
        description = param.description if hasattr(param, 'description') else ''
        param_group.attrs['description'] = description
        # TODO: param_group.attrs['available_dependencies'] = param.available_dependencies
        # TODO: Possible to store validity percentage upon name.attrs
        # TODO: Update valid param names cache rather than clearing it.
        self._valid_param_names_cache = None

    def __delitem__(self, param_name):
        '''
        Delete a parameter (and associated information) from the HDF.

        Note: Space will not be reclaimed.

        :param param_name: Parameter name to be deleted
        :type param_name: String
        '''
        if param_name in self:
            del self.hdf['series'][param_name]
            self._keys_cache.remove(param_name)
        else:
            raise KeyError("%s" % param_name)

    def delete_params(self, param_names, raise_keyerror=False):
        '''
        Calls del_param for each parameter name in list.

        Note: Space will not be reclaimed.

        :param param_name: Parameter names to be deleted
        :type param_name: List of Strings
        :param raise_keyerror: Raise KeyError if encounters a parameter that is not available
        :type raise_keyerror: Bool
        '''
        for param_name in param_names:
            try:
                del self[param_name]
            except KeyError:
                if raise_keyerror:
                    raise
                else:
                    pass  # ignore parameters that aren't available

    def valid_param_names(self):
        '''
        :returns: Only the names of valid parameters.
        :rtype: [str]
        '''
        if self._valid_param_names_cache is None:
            valid_params = []
            for param in self.keys():
                if self.hdf['series'][param].attrs.get('invalid') == 1:
                    continue
                else:
                    valid_params.append(param)
            self._valid_param_names_cache = valid_params
        return self._valid_param_names_cache

    def set_param_limits(self, name, limits):
        '''
        Stores limits for a parameter in JSON format.

        :param name: Parameter name
        :type name: str
        :param limits: Operating limits storage
        :type limits: dict
        '''
        param_group = self.get_or_create(name)
        param_group.attrs['limits'] = simplejson.dumps(limits)

    def get_param_limits(self, name, default=None):
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
        return simplejson.loads(limits) if limits else default

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
        print 'Offset:', group.attrs['supf_offset']
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
    hdf['series']['Altitude AAL'].attrs['limits'] = {'min': 0, 'max': 50000}
