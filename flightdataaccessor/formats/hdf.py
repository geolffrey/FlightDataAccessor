# Tasks to perform:
# [x] Cache parameters
# [x] Support for array slice
# [ ] Modification of "old" format file will update the old format file and parameter attributes preserving data types
#     etc.
from __future__ import division

import base64
import copy
import datetime
import functools
import math
import os
import pytz
import re
import warnings
import zlib

import h5py
import numpy as np
import simplejson
import six

from collections import defaultdict
from sortedcontainers import SortedSet

from flightdatautilities.array_operations import merge_masks
from flightdatautilities.patterns import wildcard_match

from .legacy import Compatibility
from ..datatypes.parameter import Parameter


CURRENT_VERSION = 3
LIBRARY_VERSION = (1, 10, 1)


def require_open(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.file is None:
            raise IOError('HDF file is not open')
        return func(self, *args, **kwargs)

    return wrapper


def require_rw(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.file is None:
            raise IOError('HDF file is not open')
        if self.file.mode != 'r+':
            raise IOError('Mofdification of file open in read-only mode was requested')
        return func(self, *args, **kwargs)

    return wrapper


# XXX: Should subclass container types: https://docs.python.org/2/library/collections.html#collections-abstract-base-classes
@six.python_2_unicode_compatible
class FlightDataFile(Compatibility):
    VERSION = CURRENT_VERSION
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 6}
    # attributes stored in memory
    INSTANCE_ATTRIBUTES = {
        'cache_param_list',
        'data',
        'file',
        'file_mode',
        'keys_cache',
        'parameter_cache',
        'path',
    }
    # attributyes stored in HDF file
    DYNAMIC_HDF_ATTRIBUTES = {
        'duration',
        'frequencies',
    }
    HDF_ATTRIBUTES = {
        'arinc',  # XXX
        'dependency_tree',
        'reliable_frame_counter',
        'reliable_subframe_counter',
        'superframe_present',
        'timestamp',
        'version',
        'version_analyzer',
        'version_cleanser',
        'version_converter',
    }
    ALL_ATTRIBUTES = INSTANCE_ATTRIBUTES | HDF_ATTRIBUTES

    def __init__(self, filelike, mode=None, **kwargs):
        if h5py.version.hdf5_version_tuple < LIBRARY_VERSION:
            pass  # XXX: Issue a warning?

        self.file_mode = self._parse_legacy_options(mode, **kwargs)
        if self.file_mode is None:
            self.file_mode = 'r'

        self.cache_param_list = []
        self.parameter_cache = {}
        self.keys_cache = defaultdict(SortedSet)

        created = False
        # Prepare the file for reading or writing:
        if isinstance(filelike, h5py.File):
            self.path = os.path.abspath(filelike.filename)
            self.file = filelike
            # ...
        else:
            # XXX: Handle compressed files transparently?
            self.path = os.path.abspath(filelike)
            if not os.path.exists(self.path):
                created = True
            self.file = h5py.File(filelike, mode=self.file_mode)

        # Handle backwards compatibility for older versions:
        if created:
            self.version = self.VERSION

        if self.file.attrs.get('version', 0) >= self.VERSION:
            self.data = self.file
        else:
            if 'series' not in self.file:
                # XXX: maybe we should raise an error instead, this file contains no data anyway
                self.file.create_group('series')
            self.data = self.file['series']

    def __repr__(self):
        # XXX: Make use of six.u(), etc?
        return '<%(class)s [HDF5] (%(state)s, mode %(mode)s, %(size)d bytes, %(count)d parameters) %(path)s>' % {
            'class': self.__class__.__name__,
            'count': len(self),
            'mode': self.file.mode if self.file else '',
            'path': self.path,
            'size': os.path.getsize(self.path),  # FIXME: Pretty size? OSError?
            'state': 'open' if self.file else 'closed',
        }

    def __str__(self):
        return self.__repr__().lstrip('<').rstrip('>')

    def __enter__(self):
        """Context manager API"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager API"""
        self.close()

    def __iter__(self):
        """Iterator API"""
        return iter(self.keys())

    @require_open
    def __getitem__(self, name):
        """Retrieve parameter data from the flight data file."""
        if name not in self:
            raise KeyError('Parameter not found')

        return self.get_parameter(name)

    @require_rw
    def __setitem__(self, name, parameter):
        """Update parameter data in the flight data file."""
        if name != parameter.name:
            raise ValueError('Parameter name must be the same as the key!')
        return self.set_parameter(parameter)

    @require_rw
    def __delitem__(self, name):
        """Remove a parameter from the flight data file."""
        if name not in self:
            raise KeyError('Parameter not found')

        return self.delete_parameter(name)

    @require_open
    def __getattr__(self, name):
        """Retrieve global file attribute handling special cases.

        Special behaviour: if attribute is one of the standard HDF attributes it will be returned as None if not found
        in the HDF data.
        """
        if name not in self.ALL_ATTRIBUTES | set(self.file.attrs.keys()):
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
        value = self.file.attrs.get(name)
        # Handle backwards compatibility for older versions:
        if self.file.attrs.get('version', 0) >= self.VERSION:
            return value

        # XXX move to legacy?
        if name == 'version':
            if value is None:
                value = self.file.attrs.get('hdfaccess_version')
            return 2 if value is None else value
        elif name in {'reliable_frame_counter', 'reliable_subframe_counter', 'superframe_present'}:
            return None if value is None else bool(value)
        elif name in {'dependency_tree'}:
            return simplejson.loads(zlib.decompress(base64.decodestring(value)).decode('utf-8')) if value else None
        else:
            return value

    @require_open
    def set_source_attribute(self, name, value):
        if value is not None:
            # Handle backwards compatibility for older versions:
            if self.file.attrs.get('version', 0) >= self.VERSION:
                try:
                    self.file.attrs[name] = value
                except TypeError:
                    pass
            else:
                if name in {'reliable_frame_counter', 'reliable_subframe_counter', 'superframe_present'}:
                    value = int(value)
                elif name in {'dependency_tree'}:
                    value = base64.encodestring(
                        zlib.compress(simplejson.dumps(value, separators=(',', ':')).encode('ascii')))
                elif name in {'arinc'} and value not in {'717', '767'}:
                    raise ValueError('Unknown ARINC standard: %s.' % value)
                # XXX should we attempt to store non-standard attributes in HDF file or raise an AttributeError instead?
                self.file.attrs[name] = value

        elif name in self.file.attrs:
            del self.file.attrs[name]

    def __setattr__(self, name, value):
        """Store global file attribute handling special cases.

        Attribute names are preserved depending on the format version. The attribute names and formats are converted on
        the fly.

        Special behaviour: all extra attributes are stored in HDF file.
        """
        if name in self.INSTANCE_ATTRIBUTES:
            # Handle attributes that are not stored in HDF file
            return object.__setattr__(self, name, value)

        if name in self.DYNAMIC_HDF_ATTRIBUTES:
            warnings.warn(
                '%s is calculated automatically on close(). Manually assigned value will be overwritten.' %
                name, DeprecationWarning,
            )

        return self.set_source_attribute(name, value)

    @require_rw
    def __delattr__(self, name):
        if name not in self.file.attrs:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
        del self.file.attrs[name]

    @require_open
    def __contains__(self, name):
        """Whether a parameter exists in the flight data file."""
        return name in self.keys()

    @require_open
    def __len__(self):
        """Count of parameters in the flight data file."""
        return len(self.keys())

    def keys(self, valid_only=False, subset=None):
        """Parameter group names within the series group.

        :param subset: name of a subset of parameter names to lookup.
        :type subset: str or None
        :param valid_only: whether to only lookup names of valid parameters.
        :type valid_only: bool
        :returns: sorted list of parameter names.
        :rtype: list of str
        """
        if subset and subset not in ('lfl', 'derived'):
            raise ValueError('Unknown parameter subset: %s.' % subset)
        category = subset + '_names' if subset else 'names'
        category = 'valid_' + category if valid_only else category
        if not self.keys_cache[category]:
            if subset is None and not valid_only:
                self.keys_cache[category].update(self.data.keys())
            else:
                for name in self.keys():  # (populates top-level name cache.)
                    attrs = self.data[name].attrs
                    invalid = bool(attrs.get('invalid'))
                    # XXX: assume source = 'lfl' by default?
                    source = attrs.get('source', 'lfl' if attrs.get('lfl', True) else 'derived')
                    append = not any((
                        valid_only and invalid,
                        subset == 'lfl' and source != 'lfl',
                        subset == 'derived' and source != 'derived',
                    ))
                    if append:
                        self.keys_cache[category].add(name)

        return list(self.keys_cache[category])

    def values(self):
        for name in self.data:
            yield self[name]

    def items(self):
        """Iterate over the parameters in the flight data file."""
        for name in self.data:
            yield name, self[name]

    def close(self):
        # XXX: raise IOError if no file?
        if self.file is not None and self.file.id:
            if self.file.mode == 'r+':
                self.file.flush()
                durations = np.array([p.duration for p in self.values()])
                self.duration = np.nanmax(durations) if np.any(durations) else 0
                self.frequencies = sorted({p.frequency for p in self.values()})

            self.file.close()
            self.file = None

    if six.PY2:
        iterkeys = keys
        iteritems = items
        itervalues = values

    def get_or_create(self, name):
        """Return a h5py parameter group, if it does not exist then create it too."""
        if name in self.keys():
            group = self.data[name]
        else:
            group = self.data.create_group(name)
        return group

    def load_mask(self, name):
        group = self.data[name]
        attrs = group.attrs

        submasks = {}
        if 'submasks' in attrs and 'submasks' in group:
            submask_map = attrs['submasks']
            if submask_map.strip():
                submask_map = simplejson.loads(submask_map)
                for sub_name, index in submask_map.items():
                    submasks[sub_name] = group['submasks'][slice(None), index]

            mask = merge_masks(list(submasks.values()))

            if 'mask' in group:
                old_mask = group['mask'][slice(None)]
                if np.any(mask != old_mask):
                    submasks['legacy'] = old_mask
                    mask |= old_mask
        else:
            if 'mask' in group:
                mask = group['mask']
            else:
                mask = False

        return mask, submasks

    def load_parameter(self, name, load_submasks=False):
        """Load parameter from cache or the file and store in cache"""
        if name in self.parameter_cache:
            return self.parameter_cache[name]

        group = self.data[name]
        attrs = group.attrs
        data = group['data'][:]
        name = name.split('/')[-1]

        kwargs = {}
        kwargs['frequency'] = attrs.get('frequency', 1)

        mask, submasks = self.load_mask(name)
        if load_submasks:
            kwargs['submasks'] = submasks

        array = np.ma.masked_array(data, mask=mask)

        if 'values_mapping' in attrs:
            values_mapping = attrs['values_mapping']
            if values_mapping.strip():
                mapping = simplejson.loads(values_mapping)
                kwargs['values_mapping'] = mapping

        if 'values_mapping' not in kwargs and data.dtype == np.int_:
            # Force float for non-values_mapped types.
            array = array.astype(np.float_)

        # backwards compatibility
        kwargs['source'] = attrs.get('source', 'lfl' if attrs.get('lfl', True) else 'derived')
        kwargs['offset'] = attrs.get('offset', attrs.get('supf_offset', 0))
        kwargs['unit'] = attrs.get('unit', attrs.get('units'))

        kwargs['arinc_429'] = bool(attrs.get('arinc_429', False))
        kwargs['invalid'] = bool(attrs.get('invalid', False))
        kwargs['invalidity_reason'] = attrs.get('invalidity_reason', None)
        kwargs['limits'] = attrs.get('limits', None)
        kwargs['data_type'] = attrs.get('data_type', None)
        kwargs['source_name'] = attrs.get('source_name', None)
        parameter = Parameter(name, array, **kwargs)
        # FIXME: do we want to keep this condition?
        if name in self.cache_param_list:
            self.parameter_cache[name] = parameter

        return parameter

    def get_parameter(self, name, valid_only=False, _slice=None, load_submasks=False, copy_param=True):
        """Load parameter and handle special cases"""
        if name not in self.keys(valid_only):
            raise KeyError(name)

        parameter = self.load_parameter(name, load_submasks=load_submasks)

        if _slice:
            slice_start = int((_slice.start or 0) * parameter.frequency)
            slice_stop = int((_slice.stop or parameter.array.size) * parameter.frequency)
            _slice = slice(slice_start, slice_stop)
            array = parameter.array[_slice]
            parameter = copy.deepcopy(parameter)
            parameter.array = array
        elif copy_param:
            try:
                parameter = copy.deepcopy(parameter)
            except Exception:
                attr_names = [
                    'name', 'array', 'source', 'offset', 'unit', 'arinc_429', 'invalid', 'invalidity_reason', 'limits',
                    'data_type', 'source_name', 'submasks'
                ]
                attrs = [[a, type(getattr(parameter, a)), getattr(parameter, a)] for a in attr_names]
                copy.deepcopy(attrs)
                raise

        return parameter

    @require_rw
    def set_parameter(self, parameter, save_data=True, save_mask=True, save_submasks=True):
        """Store parameter data"""
        attrs = (
            'source', 'source_name', 'data_type', 'arinc_429', 'frequency', 'offset', 'unit', 'values_mapping',
            'invalid', 'invalidity_reason', 'limits')

        if not save_mask:
            warnings.warn(
                'save_mask argument is deprecated. Parameter mask is combined from submasks to ensure consistency',
                DeprecationWarning,
            )

        if hasattr(parameter, 'validate_mask'):
            parameter.validate_mask()
        param_group = self.get_or_create(parameter.name)

        if save_data:
            if 'data' in param_group:
                del param_group['data']
            param_group.create_dataset('data', data=np.ma.getdata(parameter.array), **self.DATASET_KWARGS)

        # XXX: remove options to save masks or submasks, implement saving individual Parameter data instead
        if (save_submasks or save_mask) and getattr(parameter, 'submasks', None):
            if 'submasks' in param_group:
                del param_group['submasks']
            if 'mask' in param_group:
                del param_group['mask']

            # Get array length for expanding booleans.
            submask_length = 0
            for submask_name, submask_array in parameter.submasks.items():
                if (submask_array is None or type(submask_array) in (bool, np.bool8)):
                    continue
                submask_length = max(submask_length, len(submask_array))

            # TODO: store array mask in 'legacy' submask if it's not equivalent to the submasks

            submask_map = {}
            submask_arrays = []
            not_empty = (x for x in parameter.submasks.items() if x[1] is not None)
            for index, (submask_name, submask_array) in enumerate(not_empty):
                submask_map[submask_name] = index

                # Expand booleans to be arrays.
                if type(submask_array) in (bool, np.bool8):
                    function = np.ones if submask_array else np.zeros
                    submask_array = function(submask_length, dtype=np.bool8)
                submask_arrays.append(submask_array)

            param_group.attrs['submasks'] = simplejson.dumps(submask_map)
            param_group.create_dataset('submasks', data=np.column_stack(submask_arrays), **self.DATASET_KWARGS)

        for attr in attrs:
            if hasattr(parameter, attr):
                value = getattr(parameter, attr)
                if value is None:
                    continue

                if attr == 'values_mapping':
                    value = simplejson.dumps(value)

                param_group.attrs[attr] = value

        if parameter.name in self.cache_param_list:
            self.parameter_cache[parameter.name] = parameter

        # Update all parameter name caches with updates:
        for key, cache in self.keys_cache.items():
            cache.discard(parameter.name)
            if parameter.source == 'derived' and 'derived' in key or not parameter.source != 'derived' and 'lfl' in key:
                continue
            if parameter.invalid and key.startswith('valid'):
                continue
            self.keys_cache[key].add(parameter.name)

    @require_rw
    def delete_parameter(self, name, ignore=True):
        """Delete a parameter"""
        if name not in self:
            if ignore:
                return
            raise KeyError('Parameter not found')

        for key, cache in self.keys_cache.items():
            cache.discard(name)
        del self.data[name]

    def get_parameters(self, names=None, valid_only=False, raise_keyerror=False, _slice=None):
        """Get multiple parameters"""
        if names is None:
            names = self.keys(valid_only=valid_only)
        return {name: self.get_parameter(name, valid_only=valid_only) for name in names}

    @require_rw
    def set_parameters(self, parameters):
        """Set multiple parameters"""
        for parameter in parameters:
            self.set_parameter(parameter)

    def delete_parameters(self, names):
        """Delete multiple parameters"""
        for name in names:
            self.delete_parameter(name)

    def get_matching(self, regex_str):
        """Get parameters with names matching regex_str"""
        # XXX: is it used?
        compiled_regex = re.compile(regex_str)
        param_names = filter(compiled_regex.match, self.keys())
        return [self[param_name] for param_name in param_names]

    def trim(
            self, filename, start_offset=0, stop_offset=None, superframe_boundary=False, parameter_list=None,
            deidentify=False):
        """Create a copy of the file trimmed to given range"""
        if parameter_list is None:
            parameter_list = self.keys()
        superframe_size = 64 if self.superframe_present else 4
        if start_offset is None:
            start_offset = 0
        if stop_offset is None:
            stop_offset = self.duration
        if superframe_boundary:
            start_offset = superframe_size * math.floor(start_offset / superframe_size) if start_offset else 0
            stop_offset = superframe_size * math.ceil(stop_offset / superframe_size)
        with FlightDataFile(filename, mode='x') as new_fdf:
            for name in parameter_list:
                parameter = self.get_parameter(name, load_submasks=True)
                new_fdf.set_parameter(
                    parameter.trim(
                        start_offset=start_offset, stop_offset=stop_offset, pad=True,
                        superframe_boundary=superframe_boundary, superframe_size=superframe_size))
            for name, value in self.file.attrs.items():
                name = self.prepare_attribute_name(name)
                if name is None or name == 'version' or deidentify and name in ('aircraft_info', 'tailmark'):
                    # XXX version is set in the __init__() and should not be copied from the source
                    continue

                if name == 'duration':
                    value = stop_offset - start_offset
                elif name == 'timestamp' and self.timestamp is not None:
                    value = self.timestamp + start_offset
                new_fdf.file.attrs.create(name, value)

    @require_rw
    def concatenate(self, paths):
        """Concatenate compatible FDF files.

        ValueError is raised in case any incompatibility is found on parameter level.
        """
        for path in paths:
            with FlightDataFile(path) as fdf:
                # sanity checks
                if set(fdf.keys()) != set(self.keys()):
                    raise ValueError('The parameter list in concatenated file must be identical')
                for to_append in fdf.values():
                    parameter = self[to_append.name]
                    if not parameter.is_compatible(to_append):
                        raise ValueError('The parameters in concatenated file must be identical')

        for path in paths:
            with FlightDataFile(path) as fdf:
                for to_append in fdf.values():
                    parameter = self[to_append.name]
                    parameter.extend(to_append)
                    self[parameter.name] = parameter

    def get(self, name, default=None, **kwargs):
        """Dictionary like .get operator. Additional kwargs are passed into the get_param method."""
        try:
            return self.get_parameter(name, **kwargs)
        except KeyError:
            return default

    # XXX are the below methods used? Does it make sense to keep them in the API?
    def search(self, pattern, lfl_keys_only=False):
        """Searches for param using wildcard matching

        If a match with the regular expression is not found, then a list of params are returned that contains the
        `pattern`."""
        # XXX: is it used?
        if lfl_keys_only:
            keys = self.lfl_keys()
        else:
            keys = self.keys()
        if '(*)' in pattern or '(?)' in pattern:
            return wildcard_match(pattern, keys)
        else:
            pattern = pattern.upper()
            return sorted(k for k in keys if pattern in k.upper())

    def startswith(self, term):
        # XXX: is it used?
        """Searches for keys which start with the term. Case sensitive"""
        return sorted(x for x in self.keys() if x.startswith(term))

    def get_parameter_limits(self, name, default=None):
        """Return a parameter's operating limits stored within the groups 'limits' attribute.

        Decodes limits from JSON into dict.
        """
        limits = self.data[name].attrs.get('limits')
        return simplejson.loads(limits) if limits else default

    def get_param_arinc_429(self, name):
        """Returns a parameter's ARINC 429 flag."""
        arinc_429 = bool(self.data[name].attrs.get('arinc_429'))
        return arinc_429

    # XXX: the below methods are unbalanced: we cater for certain modifications on the parameters, but not the others
    # Maybe move to legacy instead?
    @require_rw
    def set_parameter_limits(self, name, limits):
        """Set parameter limits"""
        param_group = self.get_or_create(name)
        param_group.attrs['limits'] = simplejson.dumps(limits)

    @require_rw
    def set_parameter_invalid(self, name, reason=''):
        """Set a parameter to be invalid"""
        # XXX: originally the parameter was fully masked, should we create a submask for that?
        param_group = self.data[name]
        param_group.attrs['invalid'] = 1
        param_group.attrs['invalidity_reason'] = reason

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
