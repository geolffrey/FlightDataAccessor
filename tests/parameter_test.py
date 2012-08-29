import numpy as np
import unittest

from hdfaccess.parameter import MappedArray, Parameter


class TestMappedArray(unittest.TestCase):
    def test_create_from_list(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        a = MappedArray(values, mask=mask, values_mapping=mapping)
        self.assertTrue(a[0] == 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertTrue(a[2] == 'three')

    def test_create_from_ma(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)
        self.assertTrue(a[0] == 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertTrue(a[2] == 'three')

    def test_slice(self):
        values = [1, 2, 3]
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)
        a = a[:2]
        self.assertTrue(a[0] == 'one')
        self.assertTrue(a[1] is np.ma.masked)


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
        # FIXME: should we return None or masked values if raw value not in
        # mapping?
        self.assertEqual(p.array[2], None)
        self.assertEqual(p.raw_array[2], 3)


if __name__ == '__main__':
    unittest.main()
