#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''

import inspect
import logging
import traceback

from numpy import bool_
from numpy.ma import MaskedArray, masked, zeros

from flightdatautilities.array_operations import merge_masks

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

    For detils about numpy array subclassing see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    '''
    def __new__(subtype, *args, **kwargs):
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
        ##self.values_mapping = getattr(obj, 'values_mapping', {})

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
            self.state = {v: k for k, v in value.iteritems()}
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
        valid_states = self.values_mapping.values()
        array = zeros(len(self), dtype=bool_)
        for state in states:
            if not ignore_missing:
                if state not in valid_states:
                    raise ValueError(
                        "State '%s' is not valid. Valid states: '%s'." %
                        (state, valid_states))
            array |= self == state
        return array

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
            if hasattr(self, 'state') and other in self.state:
                other = self.state[other]
            elif hasattr(self, 'values_mapping') \
                    and other not in self.values_mapping:
                # the caller is 2 frames down on the stack
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
                other = [self.state.get(el, None if isinstance(el, basestring)
                                        else el) for el in other]
            else:
                pass  # allow equality by MaskedArray
        return other

    def __eq__(self, other):
        '''
        Allow comparison with Strings such as array == 'state'
        '''
        return MaskedArray.__eq__(self.raw, self.__coerce_type(other))

    def __ne__(self, other):
        '''
        In MappedArrays, != is always the opposite of ==
        '''
        return MaskedArray.__ne__(self.raw, self.__coerce_type(other))

    def __gt__(self, other):
        '''
        works - but comparing against string states is not recommended
        '''
        return MaskedArray.__gt__(self.raw, self.__coerce_type(other))

    def __ge__(self, other):
        return MaskedArray.__ge__(self.raw, self.__coerce_type(other))

    def __lt__(self, other):
        return MaskedArray.__lt__(self.raw, self.__coerce_type(other))

    def __le__(self, other):
        return MaskedArray.__le__(self.raw, self.__coerce_type(other))

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
            if isinstance(val, basestring):
                # expecting self[:3] = 'one'
                return super(MappedArray, self).__setitem__(
                    key, self.state[val])
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
                        if v in self.state:
                            # v is a string
                            mapped_val[i] = self.state[v]
                        elif v in self.values_mapping:
                            # v is an int
                            mapped_val[i] = v
                        elif v is masked:
                            mapped_val.data[i] = v
                            mapped_val[i] = masked
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
        self.frequency = self.sample_rate = self.hz = float(frequency)
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
        return MaskedArray(self.array.data, mask=self.submasks[submask].copy())
    
    def combine_submasks(self):
        '''
        Combine submasks into a single OR'd mask.
        
        :returns: Combined submask.
        :rtype: np.array
        '''
        if self.submasks:
            return merge_masks(self.submasks.values())
        else:
            return self.array.mask
