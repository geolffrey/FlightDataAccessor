#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''
from numpy.ma import MaskedArray, masked

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
            data=str(self),
            # WARNING: SLOW!
            sdata=str(MaskedArray([self.values_mapping.get(x, NO_MAPPING)
                                   for x in self.data], mask=self.mask)),
            mask=str(self._mask),
            fill=str(self.fill_value),
            dtype=str(self.dtype),
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

    @property
    def raw(self):
        '''
        See the raw data.
        '''
        return self.view(MaskedArray)
    
    def __eq__(self, other):
        '''
        Allow comparison with Strings such as array == 'state'
        '''
        if other in self.state:
            other = self.state[other]
        return super(MappedArray, self).__eq__(other)

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
        if not isinstance(val, int):
            # raises KeyError if mapping does not exist for val
            if isinstance(key, slice):
                mapped_val = [self.state[v] for v in val]
                return super(MappedArray, self).__setitem__(key, mapped_val)
            else:
                # single value
                return super(MappedArray, self).__setitem__(key,
                                                            self.state[val])
        return super(MappedArray, self).__setitem__(key, val)


class Parameter(object):
    def __init__(self, name, array=[], values_mapping=None, frequency=1,
                 offset=0, arinc_429=None, invalid=None, invalidity_reason=None, 
                 units=None, data_type=None, lfl=None, description=''):
        '''
        :param name: Parameter name
        :type name: String
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: Float
        :param offset: Offset in Superframe.
        :type offset: Float
        :param units:
        :type units: str
        :param description: Description of parameter.
        :type description: str
        '''
        self.name = name
        if values_mapping is not None:
            self.values_mapping = {int(k): v for k, v
                                   in values_mapping.items()}
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
        self.description = description
        self.invalid = invalid
        self.invalidity_reason = invalidity_reason

    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)
