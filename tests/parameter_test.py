import numpy as np
import unittest

from hdfaccess.parameter import MappedArray, Parameter


class TestMappedArray(unittest.TestCase):
    def test_create_from_list(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        a = MappedArray(values, mask=mask, values_mapping=mapping)
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertEqual(a[2], 'three')

    def test_create_from_ma(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertEqual(a[2], 'three')

    def test_get_slice(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)
        a = a[:2]
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        # get mapped data value
        self.assertEqual(type(a), MappedArray)
        self.assertEqual(a.state['one'], 1)
        
    def test_set_slice(self):
        values = [1, 2, 3, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)
        a[:2] = ['two', 'three'] # this will unmask second item!
        self.assertEqual(a[0], 'two')
        self.assertEqual(list(a.raw), [2, 3, 3, np.ma.masked])
        self.assertEqual(a[1], 'three') # updated value
        self.assertTrue(a[1] is not np.ma.masked) # mask is lost
        self.assertTrue(a[3] is np.ma.masked) # mask is maintained

    def test_no_mapping(self):
        # values_mapping is no longer a requirement. (check no exception raised)
        self.assertTrue(all(MappedArray(np.arange(10)).data == np.arange(10)))
        
    def test_repr(self):
        values = [1, 2, 3, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)        
        # ensure string vals is within repr
        print a.__repr__()
        self.assertTrue('one' in a.__repr__())


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
        p = Parameter('param', array=array, frequency=frequency, offset=offset,
                      arinc_429=arinc_429)
        np.testing.assert_array_equal(p.array, array)
        self.assertEqual(p.frequency, frequency)
        self.assertEqual(p.offset, offset)
        self.assertEqual(p.arinc_429, arinc_429)

    def test_multivalue_parameter(self):
        values = [1, 2, 3]
        mask = [False, True, False]
        array = np.ma.MaskedArray(values, mask)
        mapping = {1: 'One', 2: 'Two'}
        p = Parameter('param', array=array, values_mapping=mapping)
        self.assertEqual(p.array[0], 'One')
        self.assertEqual(p.raw_array[0], 1)
        self.assertTrue(p.array[1] is np.ma.masked)
        self.assertTrue(p.raw_array[1] is np.ma.masked)
        # Get a value not in the mapping
        self.assertEqual(p.array[2], '?')
        self.assertEqual(p.raw_array[2], 3)


if __name__ == '__main__':
    unittest.main()
