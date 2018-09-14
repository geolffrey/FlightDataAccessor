#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''
from __future__ import division

import inspect
import logging
import six
import traceback

from collections import defaultdict
import numpy as np
from numpy.ma import in1d, MaskedArray, masked, zeros

from flightdatautilities.array_operations import merge_masks
from .downsample import SAMPLES_PER_BUCKET, downsample


# The value used to fill in MappedArrays for keys not within values_mapping
NO_MAPPING = '?'  # only when getting values, setting raises ValueError


class MappedArray(MaskedArray):
    '''
    MaskedArray which optionally converts its values using provided mapping.
    Has a dtype of int.

    Provide keyword argument 'values_mapping' when initialising, e.g.:
        MappedArray(np.ma.arange(3, mask=[1,0,0]),
                    values_mapping={0:'zero', 2:'two'}

    Note: first argument is a MaskedArray object.

    For details about numpy array subclassing see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    '''
    def __new__(cls, *args, **kwargs):
        '''
        Create new object.

        No default mapping - use empty dictionary.
        '''
        values_mapping = kwargs.pop('values_mapping', {})
        obj = MaskedArray.__new__(MaskedArray, *args, **kwargs)
        obj.__class__ = MappedArray

        # Must occur after class change for obj.state to update!
        obj.values_mapping = values_mapping

        return obj

    def __array_finalize__(self, obj):
        '''
        Finalise the newly created object.
        '''
        super(MappedArray, self).__array_finalize__(obj)
        if not hasattr(self, 'values_mapping'):
            master_values_mapping = getattr(obj, 'values_mapping', None)
            if master_values_mapping:
                self.values_mapping = master_values_mapping

    def __array_wrap__(self, out_arr, context=None):
        '''
        Convert the result into correct type.
        '''
        return self.__apply_attributes__(out_arr)

    def __apply_attributes__(self, result):
        result.values_mapping = self.values_mapping
        return result

    def __setattr__(self, key, value):
        '''
        '''
        # Update the reverse mappings in self.state
        if key == 'values_mapping':
            self.state = defaultdict(list)
            for k, v in value.items():
                self.state[v].append(k)
            self.state = dict(self.state)
        super(MappedArray, self).__setattr__(key, value)

    def __repr__(self):
        name = 'mapped_array'
        parameters = dict(
            name=name,
            nlen=" " * len(name),
            data=self.raw,
            # WARNING: SLOW!
            sdata=MaskedArray([self.values_mapping.get(x, NO_MAPPING)
                               for x in self.data], mask=self.mask),
            mask=self._mask,
            fill=self.fill_value,
            dtype=self.dtype,
            values=self.values_mapping
        )
        short_std = """\
masked_%(name)s(values = %(sdata)s,
       %(nlen)s   data = %(data)s,
       %(nlen)s   mask = %(mask)s,
%(nlen)s    fill_value = %(fill)s,
%(nlen)svalues_mapping = %(values)s)
"""
        return short_std % parameters

    def __str__(self):
        return str(MaskedArray([self.values_mapping.get(x, NO_MAPPING)
                                for x in self.data], mask=self.mask))

    def copy(self):
        '''
        Copy custom atributes on self.copy().
        '''
        result = super(MappedArray, self).copy()
        return self.__apply_attributes__(result)

    def get_state_value(self, state):
        '''
        Return raw value for given state.
        '''
        return self.state[state]

    def any_of(self, *states, **kwargs):
        '''
        Return a boolean array containing True where the value of the
        MappedArray equals any state in states.

        :param states: List of states.
        :type states: [str]
        :param ignore_missing: If this is False, raise an exception if any of
            the states are not in the values mapping.
        :type ignore_missing: bool
        :returns: Boolean array.
        :rtype: np.ma.array(bool)
        '''
        ignore_missing = kwargs.get('ignore_missing', False)
        raw_values = []
        for state in states:
            if state not in self.state:
                if ignore_missing:
                    # do not check array as invalid states cause
                    # exception level log messages.
                    continue
                else:
                    raise ValueError(
                        "State '%s' is not valid. Valid states: '%s'." %
                        (state, self.state.keys()))
            raw_values.extend(self.state[state])
        return MaskedArray(in1d(self.raw.data, raw_values), mask=self.mask)

    def tolist(self):
        '''
        Returns the array converted into a list of states.

        :returns: A list of states.
        :rtype: list
        '''
        # OPT: values_mapping in local scope and map masked values (4x speedup)
        values_mapping = self.values_mapping.copy()
        values_mapping[None] = None
        return [values_mapping.get(x, x) for x in super(MappedArray, self).tolist()]

    @property
    def raw(self):
        '''
        See the raw data.
        '''
        return self.view(MaskedArray)

    def __coerce_type(self, other):
        '''
        Coerces an argument of unknown type into consistently numpy comparable
        type. As used by comparison methods __eq__ etc.

        e.g. 'one' -> 1, ['one', 'two'] -> [1, 2]
        '''
        try:
            if hasattr(self, 'values_mapping') and other not in self.values_mapping:
                # the caller is 2 frames statedown on the stack
                frame = inspect.stack()[2][0]
                tb = ''.join(traceback.format_list(
                    traceback.extract_stack(frame, 3)))
                logging.error(
                    tb + 'Trying to coerce value `%s` which is not a valid '
                    'state name for this mapped array', other)
        except TypeError:  # unhashable type: 'list'
            if getattr(other, 'dtype', None) == int:
                # comparable to raw array, skip past
                pass
            elif hasattr(other, '__iter__'):
                # A list of possibly mixed types is provided, convert string
                # states where possible.
                # XXX: This is only guaranteed to work if there is a single
                # state per raw value.
                other = [
                    masked if el is masked else
                    self.state[el][0] if el in self.state else None if isinstance(el, six.string_types) else el
                    for el in other]
            else:
                pass  # allow equality by MaskedArray
        return other

    def __raw_values__(self, state):
        '''
        :type state: str
        :returns: Raw values corresponding to state.
        '''
        try:
            return getattr(self, 'state', {})[state]
        except KeyError:
            if isinstance(state, six.string_types):
                raise
        except TypeError:
            pass
        return None

    def __equality__(self, other, invert=False):
        '''
        Test equality or inequality allowing for multiple raw values matching a
        state.
        '''
        raw_values = self.__raw_values__(other)
        if raw_values:
            return MaskedArray(in1d(self.raw.data, raw_values, invert=invert), mask=self.mask)
        else:
            method = MaskedArray.__ne__ if invert else MaskedArray.__eq__
            return method(self.raw, self.__coerce_type(other))

    def __relational__(self, other, method, aggregate):
        raw_values = self.__raw_values__(other)
        other = aggregate(raw_values) if raw_values else self.__coerce_type(other)
        return method(self.raw, other)

    def __eq__(self, other):
        '''
        Allow comparison with Strings such as array == 'state'
        '''
        return self.__equality__(other)

    def __ne__(self, other):
        '''
        In MappedArrays, != is always the opposite of ==
        '''
        return self.__equality__(other, invert=True)

    def __gt__(self, other):
        '''
        works - but comparing against string states is not recommended
        '''
        return self.__relational__(other, MaskedArray.__gt__, max)

    def __ge__(self, other):
        return self.__relational__(other, MaskedArray.__ge__, max)

    def __lt__(self, other):
        return self.__relational__(other, MaskedArray.__lt__, min)

    def __le__(self, other):
        return self.__relational__(other, MaskedArray.__le__, min)

    def __getitem__(self, key):
        '''
        Return mapped values.

        Note: Returns MappedArray if sliced

        Note: Returns NO_MAPPING where mapping is not available.
        Q: Shouldn't it use self.fill_value which for string types is 'N/A'
        '''
        v = super(MappedArray, self).__getitem__(key)
        if hasattr(self, 'values_mapping'):
            if isinstance(v, MappedArray):
                # Slicing or filtering
                # MappedArray()[:2] or MappedArray()[(True, False)]
                return self.__apply_attributes__(v)
            elif v is not masked:
                # Indexing for a single value
                # MappedArray()[4]
                v = self.values_mapping.get(v, NO_MAPPING)
            else:
                pass
        return v

    def __setitem__(self, key, val):
        '''
        Raises KeyError if mapping does not exist for val (unless val is
        masked)
        '''
        if val is masked or \
           isinstance(val, int) or \
           getattr(val, 'dtype', None) == int:
            # expecting:
            # self[:3] = np.ma.masked
            # self[:3] = 2
            # self[:3] = np.ma.array([2,2,2])
            return super(MappedArray, self).__setitem__(key, val)
        else:
            if isinstance(val, six.string_types):
                # expecting self[:3] = 'one'
                return super(MappedArray, self).__setitem__(
                    key, self.state[val][0])
            else:
                # expecting the following options (all the same):
                # self[:3] = ['two', 'two', 'two']
                # self[:3] = [2, 2, 2]
                # self[:3] = ['two']
                # self[:3] = [2]
                if len(val) not in (1, len(self[key])):
                    raise ValueError("Ambiguous length of values '%s' for "
                                     "array section '%s'." % (val, self[key]))
                if isinstance(val, MaskedArray) and val.dtype.kind in (
                        'i', 'f'):
                    mapped_val = val
                else:
                    mapped_val = zeros(len(val))
                    # potentially slow if val is a large array!
                    for i, v in enumerate(val):
                        if v is masked:
                            mapped_val[i] = v
                        elif v in self.state:
                            # v is a string
                            mapped_val[i] = self.state[v][0]
                        elif v in self.values_mapping:
                            # v is an int
                            mapped_val[i] = v
                        else:
                            raise KeyError(
                                "Value '%s' not in values mapping" % v)
                return super(MappedArray, self).__setitem__(key, mapped_val)


class Parameter(object):
    def __init__(self, name, array=[], values_mapping=None, frequency=1,
                 offset=0, arinc_429=None, invalid=None, invalidity_reason=None,
                 units=None, data_type=None, lfl=None, source_name=None,
                 description='', submasks=None):
        '''
        :param name: Parameter name
        :type name: String
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        :param values_mapping: Values mapping of a multi-state parameter.
        :type values_mapping: dict or None
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: float
        :param offset: The offset of the parameter in seconds within a
            superframe.
        :type offset: float
        :param arinc_429: Whether or not the parameter stores ARINC 429 data.
        :type arinc_429: bool or None
        :param invalid: Whether or not the parameter has been marked as
            invalid.
        :type invalid: bool or None
        :param invalidity_reason: The reason why the parameter was marked as
            invalid.
        :type invalidity_reason: str or None
        :param units: The unit of measurement the parameter is recorded in.
        :type units: str or None
        :param lfl: Whether or not the parameter is from the LFL or derived.
        :type lfl: bool or None
        :param source_name: The original name of the parameter.
        :type source_name: str or None
        :param description: Description of the parameter.
        :type description: str
        :param submasks: Default value is None to avoid kwarg default being mutable.
        '''
        self.name = name
        if values_mapping is not None:
            self.values_mapping = {}
            for value, state in values_mapping.items():
                try:
                    value = int(value)
                except ValueError:
                    value = float(value)
                self.values_mapping[value] = state
            self.array = MappedArray(array, values_mapping=self.values_mapping)
        else:
            self.values_mapping = None
            self.array = array
        self.raw_array = array
        # ensure frequency is stored as a float
        self.frequency = float(frequency)
        self.offset = offset
        self.arinc_429 = arinc_429
        self.units = units
        self.data_type = data_type
        self.lfl = lfl
        self.source_name = source_name
        self.description = description
        self.invalid = invalid
        self.invalidity_reason = invalidity_reason
        self.submasks = submasks or {}

    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)

    @property
    def sample_rate(self):
        return self.frequency

    @sample_rate.setter
    def sample_rate(self, v):
        self.frequency = v

    @property
    def hz(self):
        return self.frequency

    @hz.setter
    def hz(self, v):
        self.frequency = v

    def get_array(self, submask=None):
        '''
        Get the Parameter's array with an optional submask substituted for the
        mask.

        :param submask: Name of submask to return with the array.
        :type submask: str or None
        '''
        if not submask:
            return self.array
        if submask not in self.submasks:
            return None
        if isinstance(self.array, MappedArray):
            return MappedArray(self.array.data, mask=self.submasks[submask].copy(),
                               values_mapping=self.array.values_mapping)
        else:
            return MaskedArray(self.array.data, mask=self.submasks[submask].copy())

    def combine_submasks(self):
        '''
        Combine submasks into a single OR'd mask.

        :returns: Combined submask.
        :rtype: np.array
        '''
        if self.submasks:
            return merge_masks(list(self.submasks.values()))
        else:
            return self.array.mask

    def downsample(self, width, start_offset=None, stop_offset=None, mask=True):
        '''
        Downsample data in range to fit in a window of given width.
        '''
        start_ix = int(start_offset * self.hz) if start_offset else 0
        stop_ix = int(stop_offset * self.hz) if stop_offset else self.array.size
        sliced = self.array[start_ix:stop_ix]
        if not mask:
            sliced = sliced.data
        if sliced.size <= width:
            return sliced, None

        bucket_size = SAMPLES_PER_BUCKET * sliced.size // width
        if bucket_size > 1:
            downsampled = downsample(sliced, bucket_size)
            return downsampled, bucket_size
        else:
            return sliced, bucket_size

    def zoom(self, width, start_offset=0, stop_offset=None, mask=True, timestamps=False):
        '''
        Zoom out to display the data in range in a window of given width.

        Optionally combine the data with timestamp information (in miliseconds).

        This method is designed for use in data visualisation.
        '''
        downsampled, bucket_size = self.downsample(width, start_offset=start_offset, stop_offset=stop_offset, mask=mask)
        if not timestamps:
            return downsampled

        interval = 1000 * (1 if bucket_size is None else bucket_size / SAMPLES_PER_BUCKET) / self.hz
        timestamps = 1000 * (self.offset + start_offset) + interval * np.arange(downsampled.size)
        return np.ma.dstack((timestamps, downsampled))[0]
