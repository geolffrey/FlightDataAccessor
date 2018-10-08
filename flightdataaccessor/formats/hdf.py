# Tasks to perform:
# [x] Cache parameters
# [x] Support for array slice
# [ ] Modification of "old" format file will update the old format attributes preserving data types etc.

import base64
import copy
import json
import os
import pickle
import zlib

import h5py
import numpy as np
import simplejson
import six

from collections import defaultdict
from sortedcontainers import SortedSet

from flightdatautilities.array_operations import merge_masks

from .legacy import Compatibility
from ..datatypes.parameter import Parameter


CURRENT_VERSION = 3
LIBRARY_VERSION = (1, 10, 1)
DATASET_OPTIONS = {'compression': 'gzip', 'compression_opts': 6}


# XXX: Should subclass container types: https://docs.python.org/2/library/collections.html#collections-abstract-base-classes
@six.python_2_unicode_compatible
class FlightDataFile(Compatibility):
    VERSION = CURRENT_VERSION
    DATASET_KWARGS = {'compression': 'gzip', 'compression_opts': 6}

    def __init__(self, x, mode='a'):
        if h5py.version.hdf5_version_tuple < LIBRARY_VERSION:
            pass  # XXX: Issue a warning?

        self.parameter_cache = {}
        self.keys_cache = defaultdict(SortedSet)
        created = False
        # Prepare the file for reading or writing:
        if isinstance(x, h5py.File):
            self.path = os.path.abspath(x.filename)
            self.file = x
            # ...
        else:
            # XXX: Handle compressed files transparently?
            self.path = os.path.abspath(x)
            if not os.path.exists(self.path):
                created = True
            self.file = h5py.File(x, mode=mode)

        # Handle backwards compatibility for older versions:
        if created or self.file.attrs.get('version', 0) >= self.VERSION:
            self.data = self.file
            self.version = self.VERSION
        else:
            if 'series' not in self.file:
                self.file.create_group('series')
            self.data = self.file['series']
            self.version = self.file.attrs.get('version', 2)

    def __repr__(self):
        # XXX: Make use of six.u(), etc?
        return '<%(class)s [HDF5] (%(state)s, mode %(mode)s, %(size)d bytes, %(count)d parameters) %(path)s>' % {
            'class': self.__class__.__name__,
            'count': len(self),
            'mode': self.file.mode,
            'path': self.path,
            'size': os.path.getsize(self.path),  # FIXME: Pretty size? OSError?
            'state': 'open' if self.file.id else 'closed',
        }

    def __str__(self):
        return self.__repr__().lstrip('<').rstrip('>')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, name):
        """Retrieve parameter data from the flight data file."""
        return self.get_parameter(name)

    def __setitem__(self, name, parameter):
        """Update parameter data in the flight data file."""
        if name != parameter.name:
            raise ValueError('Parameter name must be the same as the key!')
        return self.set_parameter(parameter)

    def __delitem__(self, name):
        """Remove a parameter from the flight data file."""
        # XXX: Require key check?
        return self.delete_parameter(name)

    def __getattr__(self, name):
        """Retrieve global file attribute handling special cases."""
        value = self.hdf.attrs.get(name)  # FIXME: raise AttributeError if missing?
        # Handle backwards compatibility for older versions:
        if self.file.attrs.get('version', 0) >= self.VERSION:
            return value
        else:
            if name in {'reliable_frame_counter', 'reliable_subframe_counter' 'superframe_present'}:
                return None if value is None else bool(value)
            elif name in {'achieved_flight_record', 'aircraft_info'}:
                return pickle.loads(value)
            elif name in {'dependency_tree'}:
                return simplejson.loads(zlib.decompress(base64.decodestring(value)).decode('utf-8'))
            else:
                return value

    def __setattr__(self, name, value):
        """Store global file attribute handling special cases.

        Attribute names are preserved depending on the format version. The attribute names and formats are converted on
        the fly.
        """
        if name in ('file', 'path', 'data', 'keys_cache', 'parameter_cache'):
            # handle __init__ assignments
            return object.__setattr__(self, name, value)

        if value is not None:
            # Handle backwards compatibility for older versions:
            if self.file.attrs.get('version', 0) >= self.VERSION:
                self.hdf.attrs[name] = value
            else:
                if name in {'reliable_frame_counter', 'reliable_subframe_counter', 'superframe_present'}:
                    value = int(value)
                elif name in {'achieved_flight_record', 'aircraft_info'}:
                    value = pickle.dumps(value, protocol=0)
                elif name in {'dependency_tree'}:
                    value = base64.encodestring(zlib.compress(simplejson.dumps(value, separators=(',', ':')).encode('ascii')))
                elif name in {'arinc'} and value not in {'717', '767'}:
                    raise ValueError('Unknown ARINC standard: %s.' % value)
                self.hdf.attrs[name] = value

        elif name in self.hdf.attrs:
            del self.hdf.attrs[name]

    def __delattr__(self, name):
        del self.hdf.attrs[name]

    def __contains__(self, name):
        """Whether a parameter exists in the flight data file."""
        return name in self.data

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
        # backwards compatibility
        if subset == 'lfl':
            subset = 'source'

        if subset and subset not in ('source', 'derived'):
            raise ValueError('Unknown parameter subset: %s.' % subset)
        category = subset + '_names' if subset else 'names'
        category = 'valid_' + category if valid_only else category
        if not self.keys_cache[category]:
            if subset is None and not valid_only:
                self.keys_cache[category].update(self.data.keys())
            else:
                for name in self.keys():  # (populates top-level name cache.)
                    attrs = self.data[name].attrs
                    if self.version >= self.VERSION:
                        lfl = bool(attrs.get('source', True))
                    else:
                        lfl = bool(attrs.get('lfl', True))
                    append = not any((
                        valid_only and bool(attrs.get('invalid')),
                        not lfl and subset == 'source',
                        lfl and subset == 'derived',
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
        if self.file.id:
            self.file.flush()
            self.file.close()

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

        kwargs['submasks'] = {}
        if 'submasks' in attrs and 'submasks' in group:
            submask_map = attrs['submasks']
            if submask_map.strip():
                submask_map = simplejson.loads(submask_map)
                for sub_name, index in submask_map.items():
                    kwargs['submasks'][sub_name] = group['submasks'][slice(None), index]
                mask = merge_masks(list(kwargs['submasks'].values()))

            if 'mask' in group:
                old_mask = group['mask'][slice(None)]
                if np.any(mask != old_mask):
                    kwargs['submasks']['legacy'] = old_mask
                    mask |= old_mask
        else:
            if 'mask' in group:
                mask = group['mask']
            else:
                mask = np.zeros(data.size)

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
        kwargs['source'] = attrs.get('source', bool(attrs.get('lfl', True)))
        kwargs['offset'] = attrs.get('offset', attrs.get('supf_offset', 0))
        kwargs['unit'] = attrs.get('unit', attrs.get('units'))

        kwargs['arinc_429'] = bool(attrs.get('arinc_429', False))
        kwargs['invalid'] = bool(attrs.get('invalid', False))
        kwargs['invalidity_reason'] = attrs.get('invalidity_reason', None)
        kwargs['limits'] = attrs.get('limits', None)
        kwargs['data_type'] = attrs.get('data_type', None)
        kwargs['source_name'] = attrs.get('source_name', None)
        parameter = Parameter(name, array, **kwargs)
        self.parameter_cache[name] = parameter
        return parameter

    def get_parameter(self, name, valid_only=False, _slice=None, load_submasks=False, copy_param=True):
        """Load parameter and handle special cases"""
        # TODO: load_submasks
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

    def set_parameter(self, parameter):
        attrs = (
            'source', 'source_name', 'data_type', 'arinc_429', 'frequency', 'offset', 'unit', 'values_mapping',
            'invalid', 'invalidity_reason', 'limits')

        param_group = self.get_or_create(parameter.name)
        if 'data' in param_group:
            del param_group['data']

        param_group.create_dataset('data', data=parameter.array.data, **self.DATASET_KWARGS)
        if getattr(parameter, 'submasks', None):
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

            param_group.create_dataset('submasks', data=np.column_stack(submask_arrays), **self.DATASET_KWARGS)
            param_group.attrs['submasks'] = simplejson.dumps(submask_map)

        for attr in attrs:
            if hasattr(parameter, attr):
                value = getattr(parameter, attr)
                if value is None:
                    continue

                if attr == 'values_mapping':
                    value = json.dumps(value)

                param_group.attrs[attr] = value

        self.parameter_cache[parameter.name] = parameter
        self.keys_cache['names'].add(parameter.name)
        category = 'lfl_names' if parameter.source else 'derived_names'
        self.keys_cache[category].add(parameter.name)
        if not parameter.invalid:
            self.keys_cache['valid_names'].add(parameter.name)
            self.keys_cache['valid_' + category].add(parameter.name)

    def delete_parameter(self, name):
        del self.data[name]

    def get_parameters(self, names):
        return [self.get_parameter(name) for name in names]

    def set_parameters(self, parameters):
        for parameter in parameters:
            self.set_parameter(parameter)

    def delete_parameters(self, names):
        for name in names:
            self.delete_parameter(name)
