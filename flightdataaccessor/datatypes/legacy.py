from deprecation import deprecated


class Compatibility(object):
    """Support for legacy properties and methods"""

    @property
    @deprecated(details="Please use `source == 'lfl'` instead")
    def lfl(self):
        return self.source == 'lfl'

    @lfl.setter
    @deprecated(details="Please use `source = 'lfl'` instead")
    def lfl(self, s):
        self.source = 'lfl'

    @property
    @deprecated(details='Please use `source` instead')
    def units(self):
        return self.unit

    @property
    @deprecated(details='Please use `frequency` instead')
    def sample_rate(self):
        return self.frequency

    @sample_rate.setter
    @deprecated(details='Please use `frequency` instead')
    def sample_rate(self, v):
        self.frequency = v

    @property
    @deprecated(details='Please use `frequency` instead')
    def hz(self):
        return self.frequency

    @hz.setter
    @deprecated(details='Please use `frequency` instead')
    def hz(self, v):
        self.frequency = v
