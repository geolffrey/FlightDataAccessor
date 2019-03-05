import collections
import inspect
import logging
import traceback
import warnings
from collections import defaultdict

import blosc
import numpy as np
import six
from numpy.ma import MaskedArray, in1d, masked, zeros


def compress_mask(a):
    return blosc.compress_ptr(a.__array_interface__['data'][0], a.size, a.dtype.itemsize, 9, True)


def decompress_mask(d):
    return np.frombuffer(blosc.decompress(d), dtype=np.bool)


def compress_array(a):
    values_mapping = getattr(a, 'values_mapping', None)
    return (
        str(a.dtype),
        a.size,
        blosc.compress_ptr(a.__array_interface__['data'][0], a.size, a.dtype.itemsize, 9, True),
        compress_mask(a.mask) if isinstance(a.mask, np.ndarray) else a.mask,
        values_mapping,
    )


def decompress_array(d):
    dtype, size, data_blz, mask_blz, values_mapping = d
    data = np.frombuffer(blosc.decompress(data_blz), dtype=dtype)
    if isinstance(mask_blz, bytes):
        mask = decompress_mask(mask_blz)
    else:
        mask = mask_blz
    # XXX: by removing the copy() we can make the arrays immutable to avoid subtle errors
    array = np.ma.array(data, mask=mask)
    array = array.copy()
    if values_mapping:
        array = MappedArray(array, values_mapping=values_mapping)
    return array


class MaskError(ValueError):
    pass


class MappedArray(np.ma.MaskedArray):
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
    # The value used to fill in MappedArrays for keys not within values_mapping only when getting values, setting
    # raises ValueError
    NO_MAPPING = '?'

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
        super(MappedArray, self).__array_wrap__(out_arr, context)
        return self.__apply_attributes__(out_arr)

    def __apply_attributes__(self, result):
        result.values_mapping = self.values_mapping
        return result

    def __getstate__(self):
        """Store values_mapping in the pickle."""
        return {'np.ma.state': super(MappedArray, self).__getstate__(), 'values_mapping': self.values_mapping}

    def __setstate__(self, state):
        """Restore dynamic attributes from the pickle."""
        if isinstance(state, dict):
            # Numpy uses different format of state
            super(MappedArray, self).__setstate__(state['np.ma.state'])
            self.values_mapping = state['values_mapping']
        else:
            # XXX: only for tests pickled in old format!
            # TODO: make fresh pickles
            super(MappedArray, self).__setstate__(state)

    def __setattr__(self, key, value):
        # Update the reverse mappings in self.state
        if key == 'values_mapping' and value:
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
            sdata=MaskedArray([self.values_mapping.get(x, self.NO_MAPPING)
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
        return str(MaskedArray([self.values_mapping.get(x, self.NO_MAPPING)
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
        return MaskedArray(np.ma.in1d(self.raw.data, raw_values), mask=self.mask)

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

        Note: Returns self.NO_MAPPING where mapping is not available.
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
                v = self.values_mapping.get(v, self.NO_MAPPING)
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


class ParameterArray(object):
    """Array descriptior to control parameter.array assignment.

    The idea is to keep submasks in sync with the array's mask to ensure consistency."""
    def __get__(self, parameter, objtype=None):
        if parameter.compress:
            return decompress_array(parameter._array)
        else:
            return getattr(parameter, '_array', np.ma.array([]))

    def __set__(self, parameter, array):
        """Set array on parent object.

        A rebuild of parent's submasks will be triggered as a side effect."""
        if getattr(parameter, 'values_mapping', None) and not isinstance(array, MappedArray):
            values_mapping = {}
            for value, state in parameter.values_mapping.items():
                try:
                    value = int(value)
                except ValueError:
                    value = float(value)
                values_mapping[value] = state
            array = MappedArray(array, values_mapping=values_mapping)
            parameter.values_mapping = array.values_mapping
        elif isinstance(array, MappedArray):
            # normalise values mapping
            if hasattr(array, 'values_mapping'):
                parameter.values_mapping = array.values_mapping
            else:
                array.values_mapping = parameter.values_mapping
        else:
            array = np.ma.asanyarray(array)

        if np.ma.any(np.ma.getmaskarray(array)):
            try:
                parameter.validate_mask(array, parameter.submasks, strict=True)
            except MaskError:
                warnings.warn(
                    "Overriding parameter's submasks due to masked array assignment."
                    'Consider using Parameter.set_array(array, submasks) instead.',
                    DeprecationWarning
                )
                parameter.submasks = parameter.submasks_from_array(array)

        if parameter.compress:
            parameter._array = compress_array(array)
        else:
            parameter._array = array


class ParameterSubmasks(collections.MutableMapping):
    """Better control over submasks."""
    def __init__(self, *args, **kwargs):
        self.compress = kwargs.pop('compress', False)
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        if self.compress:
            return decompress_mask(self.store[key])
        else:
            return self.store[key]

    def __setitem__(self, key, value):
        if self.compress:
            self.store[key] = compress_mask(np.asanyarray(value))
        else:
            self.store[key] = np.asanyarray(value)

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
