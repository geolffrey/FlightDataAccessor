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
        p = Parameter('param')
        self.assertEqual(p.array, [])
        self.assertEqual(p.frequency, 1)
        self.assertEqual(p.offset, 0)