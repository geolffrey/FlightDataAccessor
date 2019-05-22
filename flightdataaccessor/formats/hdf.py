# Tasks to perform:
# [x] Cache parameters
# [x] Support for array slice
# [ ] Modification of "old" format file will update the old format file and parameter attributes preserving data types
#     etc.
import base64
import functools
import os
import warnings
import zlib
from collections import defaultdict

import h5py
import numpy as np
import simplejson
import six
from sortedcontainers import SortedSet

from flightdatautilities.array_operations import merge_masks
from flightdatautilities.compression import CompressedFile

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
            raise IOError('Modification of file open in read-only mode was requested')
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
        'compress',
        'compressed_file',
        'data',
        'file',
        'hdf_attributes',
        'keys_cache',
        'mode',
        'parameter_cache',
        'path',
        'start_datetime',
    }
    # attributyes stored in HDF file
    DYNAMIC_HDF_ATTRIBUTES = {
        'duration',
        'frequencies',
    }
    ALL_ATTRIBUTES = INSTANCE_ATTRIBUTES | FlightDataFormat.FDF_ATTRIBUTES

    def __init__(self, filelike, mode='r', cache_param_list=None, **kwargs):
        if h5py.version.hdf5_version_tuple < LIBRARY_VERSION:
            pass  # XXX: Issue a warning?

        self.compress = kwargs.get('compress', False)
        self.mode = mode
        self.keys_cache = defaultdict(SortedSet)
        self.parameter_cache = {}
        self.path = None
        self.file = None
        self.open(filelike, mode=mode)

        if cache_param_list is True:
            self.cache_param_list = self.keys()
        elif cache_param_list:
            self.cache_param_list = cache_param_list
        else:
            self.cache_param_list = []

    def __repr__(self):
        # XXX: Make use of six.u(), etc?
        if self.file:
            return '<%(class)s [HDF5] (%(state)s, mode %(mode)s, %(size)d bytes, %(count)d parameters) %(path)s>' % {
                'class': self.__class__.__name__,
                'count': len(self),
                'mode': self.file.mode,
                'path': self.path,
                'size': os.path.getsize(self.path),  # FIXME: Pretty size? OSError?
                'state': 'open',
            }

        return '<%(class)s [HDF5] (%(state)s, %(size)d bytes) %(path)s>' % {
            'class': self.__class__.__name__,
            'count': len(self) if self.file else 'closed',
            'path': self.path,
            'size': os.path.getsize(self.path),  # FIXME: Pretty size? OSError?
            'state': 'closed',
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

    @property
    def duration(self):
        duration = self.hdf_attributes.get('duration')
        if duration is None:
            duration = super(FlightDataFile, self).duration
        return float(duration)

    @property
    def frequencies(self):
        frequencies = self.hdf_attributes.get('frequencies')
        if frequencies is None:
            frequencies = super(FlightDataFile, self).frequencies
        return frequencies

    @require_open
    def __getattr__(self, name):
        """Retrieve file attribute.

        Special behaviour: if attribute is one of the standard HDF attributes it will be returned as None if not found
        in the HDF data.
        """
        if name not in self.ALL_ATTRIBUTES | set(self.hdf_attributes.keys()):
            raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))

        # Handle backwards compatibility for older versions:
        legacy = self.hdf_attributes.get('version', 0) >= self.VERSION
        if legacy:
            value = self.hdf_attributes.get(name)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            return value

        # XXX move to legacy?
        if name == 'version':
            value = self.hdf_attributes.get('version', self.hdf_attributes.get('hdfaccess_version'))
            return 2 if value is None else value

        name = self.source_attribute_name(name)
        if name is None:
            return None

        value = self.hdf_attributes.get(name)
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
            self.path = os.path.abspath(source)

            compressed = CompressedFile(self.path, mode=self.mode)
            if compressed.format:
                self.compressed_file = compressed
                self.path = self.compressed_file.open()

            if not os.path.exists(self.path):
                created = True
            self.file = h5py.File(source, mode=mode)
            self.mode = 'a' if mode == 'x' else mode  # save mode for reopen

        self.hdf_attributes = dict(self.file.attrs.items())

        # Handle backwards compatibility for older versions:
        if created:
            self.version = self.VERSION

        if self.hdf_attributes.get('version', 0) >= self.VERSION:
            self.data = self.file
        else:
            if 'series' not in self.file:
                # XXX: maybe we should raise an error instead, this file contains no data anyway
                self.file.create_group('series')
            self.data = self.file['series']

        return created

    def close(self):
        # XXX: raise IOError if no file?
        if self.file is not None and self.file.id:
            if self.file.mode == 'r+':
                self.file.flush()
                durations = [self.get_parameter_duration(name) for name in self]
                self.duration = np.nanmax(durations) if durations else 0
                self.frequencies = sorted({self.get_parameter_frequency(name) for name in self})

            self.file.close()
            if getattr(self, 'compressed_file', None):
                self.compressed_file.close()
                del self.compressed_file
            self.file = None
            self.data = None

    @require_rw
    def set_source_attribute(self, name, value):
        """Set attribute stored in HDF file."""
        if name != 'version':
            name = self.source_attribute_name(name)

        if name is None:
            return

        if value is not None:
            # Handle backwards compatibility for older versions:
            if self.hdf_attributes.get('version', 0) >= self.VERSION:
                try:
                    self.file.attrs[name] = value
                    self.hdf_attributes[name] = value
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
                self.hdf_attributes[name] = value

        elif name in self.hdf_attributes:
            del self.file.attrs[name]
            del self.hdf_attributes[name]

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
                    attrs = dict(self.data[name].attrs)
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

    if six.PY2:
        iterkeys = keys

    @require_rw
    def get_or_create(self, name):
        """Return a h5py parameter group, if it does not exist then create it too."""
        if name in self.keys():
            group = self.data[name]
        else:
            group = self.data.create_group(name)
        return group

    @require_open
    def load_mask(self, name, mapping=None):
        """Load masks for given parameter name."""
        group = self.data[name]

        submasks = {}
        if mapping and 'submasks' in group:
            if mapping:
                submasks_data = group['submasks'][:]
                for sub_name, index in mapping.items():
                    submasks[sub_name] = submasks_data[slice(None), index]

            mask = merge_masks(list(submasks.values()))

            if 'mask' in group:
                old_mask = group['mask'][slice(None)]
                if np.any(mask != old_mask):
                    submasks['legacy'] = old_mask
                    mask |= old_mask
        else:
            mask = group.get('mask') or False

        if isinstance(mask, (bool, np.bool8)):
            mask = np.zeros_like(group['data'], dtype=np.bool8)
        return mask, submasks

    @require_open
    def load_parameter(self, name, load_submasks=False):
        """Load parameter from cache or the file and store in cache"""
        if name in self.parameter_cache:
            return self.parameter_cache[name]

        group = self.data[name]
        attrs = dict(group.attrs)
        data = group['data'][:]
        name = name.split('/')[-1]

        kwargs = {}
        kwargs['frequency'] = attrs.get('frequency', 1)
        mapping = attrs.get('submasks')
        if mapping and mapping.strip():
            mapping = simplejson.loads(mapping)

        mask, submasks = self.load_mask(name, mapping=mapping)
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
        limits = attrs.get('limits', None)
        kwargs['limits'] = simplejson.loads(limits) if limits else {}
        kwargs['data_type'] = attrs.get('data_type', None)
        kwargs['source_name'] = attrs.get('source_name', None)
        parameter = Parameter(name, array, compress=self.compress, **kwargs)
        # FIXME: do we want to keep this condition?
        if name in self.cache_param_list:
            self.update_parameter_cache(parameter)

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
        if (save_submasks or save_mask):
            self.store_parameter_submasks(parameter, param_group=param_group)

        for attr in PARAMETER_ATTRIBUTES:
            if hasattr(parameter, attr):
                value = getattr(parameter, attr)
                if value is None:
                    continue

                if attr in ('limits', 'values_mapping'):
                    value = simplejson.dumps(value)

                param_group.attrs[attr] = value

        self.update_parameter_cache(parameter)

    def update_parameter_cache(self, parameter):
        """Update parameter in cache."""
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
        if isinstance(data, Parameter):
            # XXX: check Parameter compatibility
            data = data.array

        array = np.asarray(data)
        mask = np.ma.getmaskarray(data)
        source = self.get_parameter_source(name)
        sources = {
            'lfl': 'padding',
            'derived': 'derived',
        }
        default_submask_name = sources.get(source, 'auto')
        submask_names = self.get_parameter_submask_names(name)
        if submasks is None:
            submasks = {k: np.ma.zeros(array.size) for k in submask_names}
            submasks[default_submask_name] = mask
        else:
            if set(submasks.keys()) != set(submask_names):
                # automatically add submasks
                raise ValueError("The submasks in extensi0on don't match the ones already stored")

        # low-level data and submasks extension
        param_group = self.data[name]
        data = param_group['data']
        start_index = len(data)
        data_size = start_index + len(array)
        data.resize((data_size,))
        data[start_index:data_size] = array

        if submask_names:
            submask_arrays = [submasks[k] for k in submask_names]
            submasks_data = param_group['submasks']
            submasks_data.resize((data_size, len(submask_arrays)))
            submasks_data[start_index:data_size, ] = np.column_stack(submask_arrays)

        for key, cache in self.keys_cache.items():
            cache.discard(name)

    # XXX: the below methods are unbalanced: we cater for certain modifications on the parameters, but not the others
    # Maybe move to legacy instead?
    @require_open
    def get_parameter_attribute(self, name, attribute, default=None, transformation=None):
        """Return a parameter's attribute from parameter cache or the HDF source."""
        if name in self.parameter_cache or attribute in ('duration',):
            return getattr(self[name], attribute)

        param_group = self.data[name]
        value = param_group.attrs.get(attribute, default)
        if transformation:
            return transformation(value)
        return value

    @require_open
    def get_parameter_submask_names(self, name):
        return self.get_parameter_attribute(
            name, 'submasks', transformation=lambda x: simplejson.loads(x) if x else {})

    # XXX: move to legacy
    def get_parameter_unit(self, name):
        """Get frequency of a parameter."""
        return self.get_parameter_attribute(self, name, 'unit')

    def get_parameter_frequency(self, name):
        """Get frequency of a parameter."""
        return self.get_parameter_attribute(name, 'frequency', 1)

    def get_parameter_duration(self, name):
        """Get duration of a parameter data is seconds."""
        param_group = self.data[name]
        return param_group['data'].len() / self.get_parameter_attribute(name, 'frequency', 1)

    def get_parameter_source(self, name):
        """Get information about parameter source."""
        return self.get_parameter_attribute(name, 'source', 'lfl')

    def get_parameter_invalid(self, name):
        """Get information if parameter is invalid."""
        return self.get_parameter_attribute(name, 'invalid', False)

    def get_parameter_limits(self, name, default=None):
        """Return a parameter's operating limits stored within the groups 'limits' attribute.

        Decodes limits from JSON into dict.
        """
        if default is None:
            default = {}
        limits = self.get_parameter_attribute(
            name, 'limits', transformation=lambda x: simplejson.loads(x) if x else default)
        return limits

    def get_param_arinc_429(self, name):
        """Returns a parameter's ARINC 429 flag."""
        return self.get_parameter_attribute(name, 'arinc_429', transformation=bool)

    @require_rw
    def set_parameter_limits(self, name, limits):
        """Set parameter limits"""
        if name in self.parameter_cache:
            parameter = self[name]
            parameter.limits = limits

        param_group = self.get_or_create(name)
        param_group.attrs['limits'] = simplejson.dumps(limits)

    @require_rw
    def set_parameter_invalid(self, name, reason=''):
        """Set a parameter to be invalid"""
        # XXX: originally the parameter was fully masked, should we create a submask for that?
        if name in self.parameter_cache:
            parameter = self[name]
            parameter.invalid = True
            parameter.invalidity_reason = reason

        param_group = self.data[name]
        param_group.attrs['invalid'] = 1
        param_group.attrs['invalidity_reason'] = reason

    @require_rw
    def set_parameter_offset(self, name, offset):
        """Set a parameter offset"""
        if name in self.parameter_cache:
            parameter = self[name]
            parameter.offset = offset

        param_group = self.data[name]
        param_group.attrs['offset'] = offset
