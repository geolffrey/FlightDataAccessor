# Tasks to perform:
# [x] Cache parameters
# [x] Support for array slice
# [ ] Modification of "old" format file will update the old format file and parameter attributes preserving data types
#     etc.
from __future__ import division

import base64
import copy
import functools
import os
import warnings
import zlib

import h5py
import numpy as np
import simplejson
import six

from collections import defaultdict
from sortedcontainers import SortedSet

from flightdatautilities.array_operations import merge_masks

from ..datatypes.parameter import Parameter
from .base import FlightDataFormat


LIBRARY_VERSION = (1, 10, 1)
CURRENT_VERSION = 3
PARAMETER_ATTRIBUTES = (
    'arinc_429', 'data_type', 'frequency', 'invalid', 'invalidity_reason', 'limits', 'offset', 'source', 'source_name',
    'unit', 'values_mapping',
)


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
class FlightDataFile(FlightDataFormat):
    VERSION = CURRENT_VERSION
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 6}
    # attributes stored in memory
    INSTANCE_ATTRIBUTES = {
        'cache_param_list',
        'data',
        'file',
        'keys_cache',
        'mode',
        'parameter_cache',
        'path',
    }
    # attributyes stored in HDF file
    DYNAMIC_HDF_ATTRIBUTES = {
        'duration',
        'frequencies',
    }
    ALL_ATTRIBUTES = INSTANCE_ATTRIBUTES | FlightDataFormat.FDF_ATTRIBUTES

    def __init__(self, filelike, mode=None, **kwargs):
        super(FlightDataFormat, self).__init__()

        if h5py.version.hdf5_version_tuple < LIBRARY_VERSION:
            pass  # XXX: Issue a warning?

        mode = self._parse_legacy_options(mode, **kwargs)
        if mode is None:
            mode = 'r'

        self.parameter_cache = {}
        self.cache_param_list = []
        self.keys_cache = defaultdict(SortedSet)

        self.path = None
        self.file = None
        created = self.open(filelike, mode=mode)

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
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager API"""
        self.close()

    @require_open
    def __getattr__(self, name):
        """Retrieve file attribute.

        Special behaviour: if attribute is one of the standard HDF attributes it will be returned as None if not found
        in the HDF data.
        """
        if name not in self.ALL_ATTRIBUTES | set(self.file.attrs.keys()):
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

        # Handle backwards compatibility for older versions:
        legacy = self.file.attrs.get('version', 0) >= self.VERSION
        if legacy:
            value = self.file.attrs.get(name)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return value

        # XXX move to legacy?
        if name == 'version':
            value = self.file.attrs.get('version', self.file.attrs.get('hdfaccess_version'))
            return 2 if value is None else value

        name = self.source_attribute_name(name)
        if name is None:
            return None

        value = self.file.attrs.get(name)
        if name in {'reliable_frame_counter', 'reliable_subframe_counter', 'superframe_present'}:
            return None if value is None else bool(value)
        elif name in {'dependency_tree'}:
            return simplejson.loads(zlib.decompress(base64.decodestring(value)).decode('utf-8')) if value else None
        else:
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return value

    def open(self, source=None, mode='r'):
        # Prepare the file for reading or writing:
        if self.file is not None:
            return

        if source is None:
            # reopen the file from the stored path and mode
            return self.open(self.path, self.mode)

        created = False

        if isinstance(source, h5py.File):
            self.path = os.path.abspath(source.filename)
            self.file = source
        else:
            # XXX: Handle compressed files transparently?
            self.path = os.path.abspath(source)
            if not os.path.exists(self.path):
                created = True
            self.file = h5py.File(source, mode=mode)
            if mode == 'x':
                # save mode for reopen
                self.mode = 'a'
            else:
                self.mode = mode
        return created

    @require_rw
    def set_source_attribute(self, name, value):
        """Set attribute stored in HDF file."""
        name = self.source_attribute_name(name)
        if name is None:
            return

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
    def delete_source_attribute(self, name):
        """Delete attribute stored in HDF file."""
        if name not in self.file.attrs:
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
        del self.file.attrs[name]

    def __delattr__(self, name):
        if name in self.INSTANCE_ATTRIBUTES:
            object.__delattr__(self, name)

        self.delete_source_attribute(name)

    @require_open
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
                    if isinstance(source, bytes):
                        source = source.decode('utf8')
                    append = not any((
                        valid_only and invalid,
                        subset == 'lfl' and source != 'lfl',
                        subset == 'derived' and source != 'derived',
                    ))
                    if append:
                        self.keys_cache[category].add(name)

        return list(self.keys_cache[category])

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

    def get_or_create(self, name):
        """Return a h5py parameter group, if it does not exist then create it too."""
        if name in self.keys():
            group = self.data[name]
        else:
            group = self.data.create_group(name)
        return group

    def load_mask(self, name):
        """Load masks for given parameter name."""
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

        if isinstance(mask, (bool, np.bool8)):
            mask = np.zeros_like(group['data'], dtype=np.bool8)
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
        source = attrs.get('source')
        if isinstance(source, bytes):
            source = source.decode('utf-8')
        elif source is None:
            source = 'lfl' if attrs.get('lfl', True) else 'derived'
        kwargs['source'] = source
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

    @require_open
    def get_parameter(self, name, valid_only=False, _slice=None, copy_param=True, load_submasks=False):
        """Load parameter and handle special cases"""
        if name not in self.keys(valid_only):
            raise KeyError(name)

        parameter = self.load_parameter(name, load_submasks=load_submasks)

        if _slice:
            slice_start = int((_slice.start or 0) * parameter.frequency)
            slice_stop = int((_slice.stop or parameter.array.size) * parameter.frequency)
            return parameter.slice(slice(slice_start, slice_stop))
        elif copy_param:
            return copy.deepcopy(parameter)

        return parameter

    @require_rw
    def store_parameter_submasks(self, parameter, param_group=None):
        """Store parameter submasks in HDF5 file."""
        if param_group is None:
            param_group = self.get_or_create(parameter.name)
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

        if submask_map:
            param_group.attrs['submasks'] = simplejson.dumps(submask_map)
            param_group.create_dataset(
                'submasks', data=np.column_stack(submask_arrays), maxshape=(None, len(submask_arrays)),
                **self.DATASET_KWARGS)

    @require_rw
    def set_parameter(self, parameter, save_data=True, save_mask=True, save_submasks=True):
        """Store parameter data in HDF5 file."""
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
            param_group.create_dataset(
                'data', data=np.ma.getdata(parameter.array), maxshape=(None,), **self.DATASET_KWARGS)

        # XXX: remove options to save masks or submasks, implement saving individual Parameter data instead
        if (save_submasks or save_mask) and getattr(parameter, 'submasks', None):
            self.store_parameter_submasks(parameter, param_group=param_group)

        for attr in PARAMETER_ATTRIBUTES:
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

    def extend_parameter(self, name, data, submasks=None):
        """Extend the parameter with additional data."""
        param_group = self.data[name]

        parameter = self[name]
        parameter.extend(data, submasks)
        data_size = parameter.array.size

        # low-level data and submasks extension
        data = param_group['data']
        start_index = len(data)
        data.resize((data_size,))
        data[start_index:len(parameter.array)] = parameter.array[start_index:len(parameter.array)]

        submask_arrays = []
        not_empty = (x for x in parameter.submasks.items() if x[1] is not None)
        for index, (submask_name, submask_array) in enumerate(not_empty):
            # Expand booleans to be arrays.
            if type(submask_array) in (bool, np.bool8):
                function = np.ones if submask_array else np.zeros
                submask_array = function(data_size, dtype=np.bool8)
            submask_arrays.append(submask_array[start_index:len(parameter.array)])

        if 'submasks' not in param_group:
            self.store_parameter_submasks(parameter)
        else:
            submasks_data = param_group['submasks']
            submasks_data.resize((data_size, len(submask_arrays)))
            submasks_data[start_index:len(parameter.array), ] = np.column_stack(submask_arrays)

    # XXX: the below methods are unbalanced: we cater for certain modifications on the parameters, but not the others
    # Maybe move to legacy instead?
    @require_open
    def get_parameter_source(self, name):
        """Get information if parameter is invalid"""
        param_group = self.data[name]
        return param_group.attrs.get('source', 'lfl')

    @require_open
    def get_parameter_invalid(self, name):
        """Get information if parameter is invalid"""
        param_group = self.data[name]
        return bool(param_group.attrs.get('invalid', False))

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
