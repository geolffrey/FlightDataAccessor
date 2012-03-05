#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''

class Parameter(object):
    def __init__(self, name, array=[], frequency=1, offset=0, arinc_429=None,
                 units=None, data_type=None, description=''):
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
        self.array = array
        # ensure frequency is stored as a float
        self.frequency = self.sample_rate = self.hz = float(frequency)
        self.offset = offset
        self.arinc_429 = arinc_429
        self.units = units
        self.data_type = data_type
        self.description = description
        
    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)
    

