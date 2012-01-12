try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from hdfaccess.parameter import Parameter

class TestParameter(unittest.TestCase):
    def setUp(self):
        pass

    def test_parameter(self):
        p_name = 'param'
        p = Parameter(p_name)
        self.assertEqual(p.name, p_name)
        self.assertEqual(p.array, [])
        self.assertEqual(p.frequency, 1)
        self.assertEqual(p.offset, 0)
        self.assertEqual(p.arinc_429, None)
        array = np.ma.arange(10)
        frequency = 8
        offset = 1
        arinc_429 = True
        p = Parameter('param', array=array, np.frequency=frequency,
                      offset=offset, arinc_429=arinc_429)
        self.assertEqual(p.array, array)
        self.assertEqual(p.frequency, frequency)
        self.assertEqual(p.offset, offset)
        self.assertEqual(p.arinc_429, arinc_429)