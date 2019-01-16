"""
Base FlightDataFormat functionality.

The FlightDataFormat class defines in-memory functionality. It can be used by itself when no storage solution is
required. Please note that this is in memory representation and in case of large data frames it will use a lot of
RAM.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import datetime
import pytz
import re

import simplejson
import six

import numpy as np

from collections import defaultdict
from sortedcontainers import SortedSet

from flightdatautilities.patterns import wildcard_match

from flightdataaccessor.datatypes.parameter import Parameter

from . import compatibility
from .legacy import Compatibility


CURRENT_VERSION = 3


# XXX: Should subclass container types:
# https://docs.python.org/2/library/collections.html#collections-abstract-base-classes
@six.python_2_unicode_compatible
class FlightDataFormat(Compatibility):
    FDF_ATTRIBUTES = {
        'arinc',
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

    def __init__(self, **kwargs):
        super(FlightDataFormat, self).__init__()
        self.compress = kwargs.get('compress', False)
        self.keys_cache = defaultdict(SortedSet)
        self.data = {}

        # FDF attributes
        self.superframe_present = kwargs.get('superframe_present', False)
        self.reliable_frame_counter = kwargs.get('reliable_frame_counter', False)
        self.timestamp = None

    def __repr__(self):
        # XXX: Make use of six.u(), etc?
        return '<%(class)s [Memory] (%(count)d parameters)>' % {
            'class': self.__class__.__name__,
            'count': len(self),
        }

    def __str__(self):
        return self.__repr__().lstrip('<').rstrip('>')

    def __enter__(self):
        """Context manager API"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager API"""
        pass

    def __iter__(self):
        """Iterator API"""
        return iter(self.keys())

    def __getitem__(self, name):
        """Retrieve parameter data from the flight data format object."""
        if name not in self:
            raise KeyError('Parameter not found')

        return self.get_parameter(name)

    def __setitem__(self, name, parameter):
        """Update parameter data in the flight data format object."""
        if name != parameter.name:
            raise ValueError('Parameter name must be the same as the key!')
        return self.set_parameter(parameter)

    def __delitem__(self, name):
        """Remove a parameter from the flight data format object."""
        if name not in self:
            raise KeyError('Parameter not found')

        return self.delete_parameter(name)

    def __contains__(self, name):
        """Whether a parameter exists in the flight data format object."""
        return name in self.keys()

    def __len__(self):
        """Count of parameters in the flight data format object."""
        return len(self.keys())

    @property
    def duration(self):
        durations = np.array([p.duration for p in self.values()])
        return np.nanmax(durations) if np.any(durations) else 0

    @property
    def frequencies(self):
        return sorted({p.frequency for p in self.values()})

    def keys(self, valid_only=False, subset=None):
        """Parameter group names within the series group.

        The supported subsets: 'lfl' and 'derived'."""
        if subset and subset not in ('lfl', 'derived'):
            raise ValueError('Unknown parameter subset: %s.' % subset)
        category = subset + '_names' if subset else 'names'
        category = 'valid_' + category if valid_only else category
        if not self.keys_cache[category]:
            if subset is None and not valid_only:
                self.keys_cache[category].update(self.data.keys())
            else:
                for name in self.keys():  # (populates top-level name cache.)
                    invalid = self.get_parameter_invalid(name)
                    source = self.get_parameter_source(name)
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
        """Iterate over the parameters in the flight data format object."""
        for name in self.data:
            yield name, self[name]

    if six.PY2:
        iterkeys = keys
        iteritems = items
        itervalues = values

    def load_parameter(self, name, **kwargs):
        """Load parameter"""
        return self.data[name]

    def get_parameter(self, name, valid_only=False, _slice=None, copy_param=True, **kwargs):
        """Load parameter and handle special cases"""
        if name not in self.keys(valid_only):
            raise KeyError(name)

        parameter = self.load_parameter(name, **kwargs)
        if isinstance(parameter.array.mask, np.bool8):
            # ensure the returned mask is always an array
            parameter.array.mask = np.ma.getmaskarray(parameter.array)

        if _slice:
            slice_start = int((_slice.start or 0) * parameter.frequency)
            slice_stop = int((_slice.stop or parameter.array.size) * parameter.frequency)
            _slice = slice(slice_start, slice_stop)
            array = parameter.array[_slice]
            parameter = copy.deepcopy(parameter)
            parameter.array = array
        elif copy_param:
            parameter = copy.deepcopy(parameter)

        return parameter

    def set_parameter(self, parameter, **kwargs):
        """Store parameter data"""
        if not parameter.invalid and hasattr(parameter, 'validate_mask'):
            parameter.validate_mask()

        self.data[parameter.name] = parameter

        # Update all parameter name caches with updates:
        for key, cache in self.keys_cache.items():
            cache.discard(parameter.name)
            if parameter.source == 'derived' and 'derived' in key or not parameter.source != 'derived' and 'lfl' in key:
                continue
            if parameter.invalid and key.startswith('valid'):
                continue
            self.keys_cache[key].add(parameter.name)

    def delete_parameter(self, name, ignore=True):
        """Delete a parameter."""
        if name not in self:
            if ignore:
                return
            raise KeyError('Parameter not found')

        for key, cache in self.keys_cache.items():
            cache.discard(name)
        del self.data[name]

    def get_parameters(self, names=None, valid_only=False, raise_keyerror=False, _slice=None):
        """Get multiple parameters."""
        if names is None:
            names = self.keys(valid_only=valid_only)
        return {name: self.get_parameter(name, valid_only=valid_only) for name in names}

    def set_parameters(self, parameters):
        """Set multiple parameters."""
        for parameter in parameters:
            self.set_parameter(parameter)

    def delete_parameters(self, names):
        """Delete multiple parameters."""
        for name in names:
            self.delete_parameter(name)

    def get_matching(self, regex_str):
        """Get parameters with names matching regex_str."""
        # XXX: is it used?
        compiled_regex = re.compile(regex_str)
        param_names = filter(compiled_regex.match, self.keys())
        return [self[param_name] for param_name in param_names]

    def trim(
            self, target=None, start_offset=0, stop_offset=None, superframe_boundary=True, parameter_list=None,
            deidentify=False):
        """Create a copy of the object trimmed to given range"""
        if target is None:
            target = self.__class__

        if parameter_list is None:
            parameter_list = self.keys()

        superframe_size = 64 if superframe_boundary and self.superframe_present else 4

        with compatibility.open(target, mode='x') as new_fdf:
            for name in parameter_list:
                parameter = self.get_parameter(name, copy_param=False)
                clone = parameter.trim(
                    start_offset=start_offset, stop_offset=stop_offset, pad_subframes=superframe_size)
                new_fdf[name] = clone
            for name in self.FDF_ATTRIBUTES:
                if name is None or name == 'version' or deidentify and name in ('aircraft_info', 'tailmark'):
                    # XXX version is set in the __init__() of the new object and should not be copied from the source
                    continue
                elif name == 'timestamp' and self.timestamp is not None:
                    value = self.timestamp + start_offset
                else:
                    value = getattr(self, name, None)
                if value:
                    setattr(new_fdf, name, value)

        return new_fdf

    def concatenate(self, sources):
        """Concatenate compatible FDF objects.

        ValueError is raised in case any incompatibility is found on parameter level.
        """
        for source in sources:
            with compatibility.open(source) as fdf:
                # sanity checks
                if set(fdf.keys()) != set(self.keys()):
                    raise ValueError('The parameter list in concatenated object must be identical')
                for to_append in fdf.values():
                    parameter = self[to_append.name]
                    if not parameter.is_compatible(to_append):
                        raise ValueError('The parameters in concatenated object must be identical')

        for source in sources:
            with compatibility.open(source) as fdf:
                for to_append in fdf.values():
                    self.extend_parameter(to_append.name, to_append)

    def get(self, name, default=None, **kwargs):
        """Dictionary like .get operator. Additional kwargs are passed into the get_parameter() method."""
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

    def get_parameter_source(self, name):
        """Get parameter source metadata"""
        return self[name].source

    def get_parameter_invalid(self, name):
        """Get information if parameter is invalid"""
        return self[name].invalid

    def get_parameter_limits(self, name, default=None):
        """Return a parameter's operating limits stored within the groups 'limits' attribute.

        Decodes limits from JSON into dict.
        """
        if default is None:
            default = {}
        limits = self[name].limits
        return simplejson.loads(limits) if limits else default

    def get_param_arinc_429(self, name):
        """Returns a parameter's ARINC 429 flag."""
        arinc_429 = bool(self[name].arinc_429)
        return arinc_429

    def extend_parameter(self, name, array, submasks=None):
        """Extend the parameter with additional data."""
        parameter = self[name]
        parameter.extend(array, submasks)
        self[parameter.name] = parameter

    # XXX: the below methods are unbalanced: we cater for certain modifications on the parameters, but not the others
    # Maybe move to legacy instead?
    def set_parameter_limits(self, name, limits):
        """Set parameter limits"""
        parameter = self.get_parameter(name)
        parameter.limits = simplejson.dumps(limits)

    def set_parameter_invalid(self, name, reason=''):
        """Set a parameter to be invalid"""
        # XXX: originally the parameter was fully masked, should we create a submask for that?
        parameter = self.get_parameter(name)
        parameter.invalid = 1
        parameter.invalidity_reason = reason

    @property
    def start_datetime(self):
        '''The start datetime of the data stored within the FDF object.

        Convert the root-level 'timestamp' attribute from a timestamp to a datetime.
        '''
        timestamp = self.timestamp
        if timestamp:
            return datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)
        else:
            return None

    @start_datetime.setter
    def start_datetime(self, start_datetime):
        """Convert start_datetime to a timestamp and save as 'timestamp' attribute.

        If start_datetime is None the 'timestamp' attribute will be deleted."""
        if start_datetime is None:
            timestamp = None
        else:
            if isinstance(start_datetime, datetime.datetime):
                epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
                timestamp = (start_datetime - epoch).total_seconds()
            else:
                timestamp = start_datetime

        self.timestamp = timestamp

    def upgrade(self, target=None):
        """Return a copy of self."""
        if target is None:
            target = type(self)

        return self.trim(target)

    # XXX: do we need it? I's used by "access attribute" methods (inconsistently) and it's use cases are questionable
    # (access an attribute on a parameter before it is properly instantiated)
    def get_or_create(self, name):
        """Return a Parameter, if it does not exist then create it too."""
        if name in self.keys():
            parameter = self.data[name]
        else:
            parameter = Parameter(name, compress=self.compress)
            self.set_parameter(parameter)
        return parameter
