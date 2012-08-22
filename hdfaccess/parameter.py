#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''
from numpy.ma import MaskedArray, masked


class MappedArray(MaskedArray):
    '''
    MaskedArray which optionally converts its values using provided mapping.

    For detils about numpy array subclassing see
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    '''
    def __new__(subtype, *args, **kwargs):
        '''
        Create new object.
        '''
        values_mapping = kwargs.pop('values_mapping', {})
        obj = MaskedArray.__new__(MaskedArray, *args, **kwargs)
        obj.values_mapping = values_mapping
        obj.rev_values_mapping = {
            v: k for k, v in obj.values_mapping.iteritems()}
        obj.__class__ = MappedArray
        return obj

    def __array_finalize__(self, obj):
        '''
        Finalise the newly created object.
        '''
        self.values_mapping = getattr(obj, 'values_mapping', None)
        self.rev_values_mapping = getattr(obj, 'rev_values_mapping', None)

    def __array_wrap__(self, out_arr, context=None):
        '''
        Convert the result into correct type.
        '''
        result = out_arr.view(MappedArray)
        return self.__apply_attributes__(result)

    def __apply__attributes__(self, result):
        result.values_mapping = self.values_mapping
        result.rev_values_mapping = self.rev_values_mapping
        return result

    def copy(self):
        '''
        Copy custom atributes on self.copy().
        '''
        result = super(MappedArray, self).copy()
        return self.__apply_attributes__(result)

    @property
    def raw(self):
        '''
        See the raw data.
        '''
        return self.view(MaskedArray)

    def __getitem__(self, key):
        '''
        Return mapped values.
        '''
        v = super(MappedArray, self).__getitem__(key)
        if self.values_mapping:
            if isinstance(key, slice):
                data = [self.values_mapping.get(x, None) for x in v.data]
                v = MaskedArray(data, v.mask)
            else:
                if v is not masked:
                    v = self.values_mapping.get(v, None)
        return v

    # def __setitem__(self, key, val):
    #     if isinstance(key, slice) and self.values_mapping:
    #     v = self.rev_values_mapping.get(val, val)
    #     return super(MappedArray, self).__setitem__(key, v)


class Parameter(object):
    def __init__(self, name, array=[], values_mapping=None, frequency=1,
                 offset=0, arinc_429=None, units=None, data_type=None,
                 description=''):
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
        self.description = description

    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)
