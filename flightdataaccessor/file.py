from __future__ import print_function

import base64
import collections
import logging
import h5py
import numpy as np
import os
import pickle
import re
import simplejson
import six
import zlib
import pytz

from collections import defaultdict
from copy import deepcopy
from datetime import datetime

from sortedcontainers import SortedSet

from flightdatautilities.compression import CompressedFile, ReadOnlyCompressedFile
from flightdatautilities.filesystem_tools import pretty_size
from flightdatautilities.patterns import wildcard_match

from flightdataaccessor.datatypes.parameter import Parameter


from .formats import hdf

HDFACCESS_VERSION = hdf.CURRENT_VERSION


hdf_file = hdf.FlightDataFile


class hdf_file_legacy(object):    # rare case of lower case?!
    """ usage example:
    with hdf_file('path/to/file.hdf5') as hdf:
        print(hdf['Altitude AAL']['data'][:20])

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
        Displays open or closed state (assists reopening)
        '''
        state = 'Open' if self.hdf.id else 'Closed'
        try:
            size = pretty_size(os.path.getsize(self.file_path))
        except OSError:
            size = 'unknown size'
        return '<%s HDF5 (%s, %d parameters) %s>' % (state, size, len(self), self.file_path)

    def __str__(self):
        return self.__repr__().lstrip('<').rstrip('>')

    def __init__(self, file_path_or_obj, cache_param_list=False, create=False,
                 read_only=False):
        '''
        Opens an HDF file (or accepts and already open h5py.File object) - will
        create if does not exist if create=True!

        :param cache_param_list: Names of parameters to cache where accessed. A value of True will result in all parameters to be cached.
        :type cache_param_list: [str] or bool
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
            if read_only:
                self.compressor = ReadOnlyCompressedFile(self.file_path)
                mode = 'r'
            else:
                self.compressor = CompressedFile(self.file_path)
                mode = 'a'
            uncompressed_path = self.compressor.load()
            self.hdf = h5py.File(uncompressed_path, mode=mode)

        self.hdfaccess_version = self.hdf.attrs.get('hdfaccess_version', 1)
        if hdf_exists:
            # default version is 1
            assert self.hdfaccess_version == HDFACCESS_VERSION
        else:
            # just created this file, add the current version
            self.hdf.attrs['hdfaccess_version'] = HDFACCESS_VERSION

        if 'series' not in self.hdf.keys():
            # The 'series' group is required for storing parameters.
            self.hdf.create_group('series')
        # cache keys as accessing __iter__ on hdf groups is v.slow
        self._cache = defaultdict(SortedSet)
        # cache parameters that are used often
        self._params_cache = {}
        # this is the list of parameters to cache
        if cache_param_list is True:
            cache_param_list = self.keys()
        elif cache_param_list is False:
            cache_param_list = []
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

    def items(self):
        for param_name in self.keys():
            yield param_name, self[param_name]

    def values(self):
        for param_name in self.keys():
            yield self[param_name]

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

    def keys(self, valid_only=False, subset=None):
        '''
        Parameter group names within the series group.

        :param subset: name of a subset of parameter names to lookup.
        :type subset: str or None
        :param valid_only: whether to only lookup names of valid parameters.
        :type valid_only: bool
        :returns: sorted list of parameter names.
        :rtype: list of str
        '''
        if subset and subset not in ('lfl', 'derived'):
            raise ValueError('Unknown parameter subset: %s.' % subset)
        key = subset + '_names' if subset else 'names'
        key = 'valid_' + key if valid_only else key
        if not self._cache[key]:
            series = self.hdf['series']
            if subset is None and not valid_only:
                self._cache[key].update(series.keys())
            else:
                for name in self.keys():  # (populates top-level name cache.)
                    attrs = series[name].attrs
                    lfl = bool(attrs.get('lfl'))
                    append = not any((
                        valid_only and bool(attrs.get('invalid')),
                        not lfl and subset == 'lfl',
                        lfl and subset == 'derived',
                    ))
                    if append:
                        self._cache[key].add(name)
        return list(self._cache[key])

    if six.PY2:
        iteritems = items
        iterkeys = keys
        itervalues = values

    # TODO: These are deprecated and should be removed!
    get_param_list = lambda self: self.keys()
    valid_param_names = lambda self: self.keys(valid_only=True)
    valid_lfl_param_names = lambda self: self.keys(valid_only=True, subset='lfl')
    lfl_keys = lambda self: self.keys(subset='lfl')
    derived_keys = lambda self: self.keys(subset='derived')

    def close(self):
        self.hdf.flush()  # Q: required?
        self.hdf.close()
        self.compressor.save()

    # HDF Attribute properties
    ############################################################################

    @property
    def analysis_version(self):
        '''
        Accessor for the root-level 'analysis_version' attribute.

        :returns: Version of the FlightDataAnalyzer which processed this HDF file.
        :rtype: str or None
        '''
        return self.hdf.attrs.get('analysis_version')

    @analysis_version.setter
    def analysis_version(self, analysis_version):
        '''
        Mutator for the root-level 'analysis_version' attribute. If version is None the
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
    def arinc(self):
        '''
        Accessor for the root-level 'arinc' attribute.

        :returns: ARINC flight data standard - either '717' or '767'.
        :rtype: str or None
        '''
        return self.hdf.attrs.get('arinc')

    @arinc.setter
    def arinc(self, arinc):
        '''
        Mutator for the root-level 'version' attribute. If version is None the
        'version' attribute will be deleted.

        :param arinc: ARINC flight data standard - either '717' or '767'.
        :type arinc: str
        :rtype: None
        '''
        if arinc is None:  # Cannot store None as an HDF attribute.
            if 'arinc' in self.hdf.attrs:
                del self.hdf.attrs['arinc']
        else:
            if arinc not in ('717', '767'):
                raise ValueError("Unknown ARINC standard '%s'." % arinc)
            self.hdf.attrs['arinc'] = arinc

    @property
    def dependency_tree(self):
        '''
        Accessor for the root-level 'dependency_tree' attribute.

        Load into a networkx graph like this:

        gr_st = json.loads(self.dependency_tree)

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
                    zlib.compress(simplejson.dumps(dependency_tree).encode('ascii')))

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
    def frequencies(self):
        '''
        Accessor for the root-level 'frequencies' attribute.

        :returns: Parameter frequencies stored within the file.
        :rtype: str or None
        '''
        # XXX: This attribute is only set by the converter which does so
        #      inconsistently, and as additional parameters are added by the
        #      analyzer which doesn't update this, the value can be incorrect.
        value = self.hdf.attrs.get('frequencies')
        if isinstance(value, collections.Iterable):  # Not the best check, but close enough.
            return value
        # XXX: Fallback to fetching the set of frequencies from all parameters.
        #      When we implement the next version of this, we could keep a flag
        #      to determine when something has changed and then properly update
        #      the attribute prior to the file being closed.
        return sorted({float(x.attrs['frequency']) for x in six.itervalues(self.hdf['series'])})

    @frequencies.setter
    def frequencies(self, frequencies):
        '''
        Mutator for the root-level 'version' attribute. If version is None the
        'version' attribute will be deleted.

        :param arinc: ARINC flight data standard - either '717' or '767'.
        :type arinc: str
        :rtype: None
        '''
        if frequencies is None:  # Cannot store None as an HDF attribute.
            if 'frequencies' in self.hdf.attrs:
                del self.hdf.attrs['frequencies']
        else:
            self.hdf.attrs['frequencies'] = frequencies

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
        if reliable_frame_counter is None and 'reliable_frame_counter' in self.hdf.attrs:
            del self.hdf.attrs['reliable_frame_counter']
        else:
            self.hdf.attrs['reliable_frame_counter'] = 1 if reliable_frame_counter else 0

    @property
    def reliable_subframe_counter(self):
        '''
        Accessor for the root-level 'reliable_subframe_counter' attribute.

        :rtype: bool or None
        '''
        reliable_subframe_counter = self.hdf.attrs.get('reliable_subframe_counter')
        return bool(reliable_subframe_counter) if reliable_subframe_counter is not None else None

    @reliable_subframe_counter.setter
    def reliable_subframe_counter(self, reliable_subframe_counter):
        '''
        Mutator for the root-level 'reliable_subframe_counter' attribute.
        If reliable_subframe_counter is None the 'reliable_subframe_counter' attribute
        will be deleted.

        :param reliable_subframe_counter: Flag indicating whether frame counter is reliable
        :type reliable_subframe_counter: bool
        :rtype: None
        '''
        if reliable_subframe_counter is None and 'reliable_subframe_counter' in self.hdf.attrs:
            del self.hdf.attrs['reliable_subframe_counter']
        else:
            self.hdf.attrs['reliable_subframe_counter'] = 1 if reliable_subframe_counter else 0

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
        if timestamp:
            return datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
        else:
            return None

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
                epoch = datetime(1970, 1, 1, tzinfo=pytz.utc)
                timestamp = (start_datetime - epoch).total_seconds()
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
            param_names = self.keys(valid_only)
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

    def get_param(self, name, valid_only=False, _slice=None, load_submasks=False, copy_param=True):
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
        :param load_submasks: Load parameter submasks into a submasks dictionary.
        :type load_submasks: bool
        :param copy_param: Return a copy of the parameter if found in cache.
        :type copy_param: bool
        :returns: Parameter object containing HDF data and attrs.
        :rtype: Parameter

        Warning: caching in this method does not work with slicing. If you
        want to slice the data and use parameter cache at the same time, don't
        pass _slice to the method (will cause the cache to be skipped), use
        copy_param=False and slice the array in your procedure instead. The
        overhead of using deepcopy on huge arrays is substantial and the tests
        of generalised slicing of the cached arrays in this method proved less
        effective than specialised slicing of the returned cached data.

        The other problem is loading of submasks: this feature is off by
        default but the parameter will be cached with the submasks if that's
        what the user requested on first fetch. If the requested submasks are
        missing in the cache, the parameter will be refetched from the disk and
        stored in the cache with the submasks. This bit can be optimised but we
        need to consider what are the use cases of submasks caching.
        '''
        # XXX: Can we avoid this extra check and rely on h5py throwing KeyError?
        if name not in self.keys(valid_only):
            raise KeyError("%s" % name)

        if name in self._params_cache:
            if not _slice:
                logging.debug("Retrieving param '%s' from HDF cache", name)
                # XXX: Caching breaks later loading of submasks!
                parameter = self._params_cache[name]
                if parameter.submasks or not load_submasks:
                    return deepcopy(parameter) if copy_param else parameter
            logging.debug(
                'Skipping returning parameter `%s` from cache as slice of '
                'data was requested.', name)
        group = self.hdf['series'][name]
        attrs = group.attrs
        data = group['data']
        mask = group.get('mask', False)

        kwargs = {}

        frequency = attrs.get('frequency', 1)
        kwargs['frequency'] = frequency

        if _slice:
            slice_start = int((_slice.start or 0) * frequency)
            slice_stop = int((_slice.stop or len(data)) * frequency)
            _slice = slice(slice_start, slice_stop)
            data = data[_slice]
            mask = mask[_slice] if mask else mask

        if load_submasks and 'submasks' in attrs and 'submasks' in group.keys():
            kwargs['submasks'] = {}
            submask_map = attrs['submasks']
            if submask_map.strip():
                submask_map = simplejson.loads(submask_map)
                for sub_name, index in submask_map.items():
                    kwargs['submasks'][sub_name] = group['submasks'][_slice or slice(None), index]

        array = np.ma.masked_array(data, mask=mask)

        if 'values_mapping' in attrs:
            values_mapping = attrs['values_mapping']
            if values_mapping.strip():
                mapping = simplejson.loads(values_mapping)
                kwargs['values_mapping'] = mapping

        if 'values_mapping' not in kwargs and data.dtype == np.int_:
            # Force float for non-values_mapped types.
            array = array.astype(np.float_)

        # Backwards compatibility. Q: When can this be removed?
        if 'supf_offset' in attrs:
            kwargs['offset'] = attrs['supf_offset']
        if 'invalid' in attrs:
            kwargs['invalid'] = attrs['invalid']
            if kwargs['invalid'] and 'invalidity_reason' in attrs:
                kwargs['invalidity_reason'] = attrs['invalidity_reason']

        keys = ('arinc_429', 'data_type', 'description', 'source', 'source_name', 'units')
        kwargs.update((key, attrs[key]) for key in keys if key in attrs)

        parameter = Parameter(name, array, **kwargs)
        # add to cache if required
        if name in self.cache_param_list:
            if not _slice:
                self._params_cache[name] = parameter
            else:
                logging.debug('Skipping saving parameter `%s` to cache as '
                              'slice of data was requested.', name)

        return parameter

    def get(self, name, default=None, **kwargs):
        """
        Dictionary like .get operator. Additional kwargs are passed into the
        get_param method.

        Makes no distinction on valid or invalid parameters that are requested.
        """
        try:
            return self.get_param(name, **kwargs)
        except KeyError:
            return default

    def get_or_create(self, name):
        '''
        Return a h5py parameter group, if it does not exist then create it too.
        '''
        # Either get or create parameter.
        if name in self.keys():
            group = self.hdf['series'][name]
        else:
            self._cache['names'].add(name)  # Update cache.
            group = self.hdf['series'].create_group(name)
            group.attrs['name'] = str(name)  # Fails to set unicode attribute.
        return group

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

        The following attributes are stored on the parameter:
         - arinc_429: Whether or not this parameter is recorded using the Digital Information Transfer System (DITS) self-clocking, self-synchronizing data bus protocol.
         - data_type: The data type of the parameter as defined within the Logical Frame Layout.
         - description: Optional description of the parameter.
         - frequency: Frequency/sample rate of the parameter's data.
         - lfl: Whether this parameter was defined in the Logical Frame Layout or derived within the Flight Data Analyser.
         - source_name: The name of this parameter within the original frame documentation.
         - submasks: The storage configuration of array masks which correspond to this parameter stored in JSON in the format {"submask_name": array_index_within_submasks_dataset}, e.g. {"padding": 0}.
         - supf_offset: The offset of this parameter in seconds within the frame or superframe.
         - units: The unit this parameter is measured in.

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
        if hasattr(param, 'values_mapping') and param.values_mapping is not None:
            param_group.attrs['values_mapping'] = simplejson.dumps(
                param.values_mapping)
        if hasattr(param, 'source_name') and param.source_name:
            param_group.attrs['source_name'] = param.source_name
        description = param.description if hasattr(param, 'description') else ''
        param_group.attrs['description'] = description
        # TODO: param_group.attrs['available_dependencies'] = param.available_dependencies
        # TODO: Possible to store validity percentage upon name.attrs

        # Update all parameter name caches with updates:
        for key, cache in self._cache.items():
            cache.discard(param.name)
            if not cache:
                continue  # don't add to the cache if it is empty.
            if param.lfl and 'derived' in key or not param.lfl and 'lfl' in key:
                continue
            if getattr(param, 'invalid', False) and key.startswith('valid'):
                continue
            self._cache[key].add(param.name)

        # Invalidate the parameter cache
        self._params_cache.pop(param.name, None)

    def set_invalid(self, name, reason=''):
        '''
        Set a parameter to be invalid and fully masked.

        :param name: Parameter name to be set invalid
        :type name: str
        :param reason: Optional invalidity reason.
        :type reason: str
        :rtype: None
        '''
        param_group = self.hdf['series'][name]
        param_group.attrs['invalid'] = 1
        param_group.attrs['invalidity_reason'] = reason
        param_group['mask'][:] = True

    def __delitem__(self, name):
        '''
        Delete a parameter (and associated information) from the HDF.

        Note: Space will not be reclaimed.

        :param param_name: Parameter name to be deleted
        :type param_name: str
        :rtype: None
        '''
        del self.hdf['series'][name]
        for cache in self._cache.values():
            if name in cache:
                cache.remove(name)

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
        '''
        limits = self.hdf['series'][name].attrs.get('limits')
        return simplejson.loads(limits) if limits else default

    def get_param_arinc_429(self, name):
        '''
        Returns a parameter's ARINC 429 flag.

        :param name: Parameter name
        :type name: str
        :returns: Parameter ARINC 429 flag.
        :rtype: bool
        '''
        arinc_429 = bool(self.hdf['series'][name].attrs.get('arinc_429'))
        return arinc_429

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
    for group_name, group in hdf_file.hdf['series'].items():
        print('[%s]' % group_name)
        print('Frequency:', group.attrs['frequency'])
        print(group.attrs.items())
        print('Offset:', group.attrs['supf_offset'])
        print('Number of recorded values:', len(group['data']))


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
