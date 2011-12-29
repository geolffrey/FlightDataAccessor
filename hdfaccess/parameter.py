#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''

import numpy as np

from analysis.library import align, value_at_time # WARNING: Circular dependency on repository


class Parameter(object):
    def __init__(self, name, array=np.ma.masked_array([]), frequency=1, offset=0):
        '''
        :param name: Parameter name
        :type name: String
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: Float
        :param offset: Offset in Frame.
        :type offset: Float
        '''
        self.name = name
        self.array = array
        # ensure frequency is stored as a float
        self.frequency = self.sample_rate = self.hz = float(frequency)
        self.offset = offset
        
    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)
    
    def get_aligned(self, param):
        aligned_array = align(self, param)
        return self.__class__(self.name, array=aligned_array,
                              frequency=param.frequency, offset=param.offset)
    
    def at(self, secs):
        """
        Interpolates to retrieve the most accurate value.
        
        :param secs: time delta from start of data in seconds
        :type secs: float or timedelta
        """
        try:
            # get seconds from timedelta
            secs = float(secs.total_seconds)
        except AttributeError:
            # secs is a float
            secs = float(secs)
        return value_at_time(self.array, self.frequency, self.offset, secs)
        
P = Parameter # shorthand
