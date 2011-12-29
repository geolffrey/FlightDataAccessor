try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np

from hdfaccess.parameter import Parameter

class TestParameter(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_parameter_at(self):
        # using a plain range as the parameter array, the results are equal to 
        # the index used to get the value (cool)
        spd = Parameter('Airspeed', np.ma.array(range(20)), 2, 0.75)
        self.assertEqual(spd.at(0.75), 0) # min val possible to return
        self.assertEqual(spd.at(1.75), 1*2) # one second in (*2Hz)
        self.assertEqual(spd.at(2.5), 1.75*2) # interpolates
        self.assertEqual(spd.at(9.75), 9*2) # max val possible to return
        
        #Q: Is this the desired behaivour to give an IndexError at second 0?
        # ... it would be misleading to use interpolation.
        self.assertRaises(ValueError, spd.at, 0)
        self.assertRaises(ValueError, spd.at, 11)
        
