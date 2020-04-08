import pickle
import unittest

import numpy as np

from flightdataaccessor.datatypes.array import (
    ParameterSubmasks, compress_array, compress_ndarray, decompress_array, decompress_ndarray,
)
from flightdataaccessor.datatypes.parameter import MappedArray, Parameter


class TestCompression(unittest.TestCase):
    def test_compress_mask(self):
        mask = (np.arange(100) % 7).astype(np.bool)
        compressed = compress_ndarray(mask)
        self.assertTrue(np.all(mask == decompress_ndarray(compressed)))

    def test_compress_numeric_array(self):
        array = np.ma.arange(100)
        compressed = compress_array(array)
        self.assertTrue(np.all(array == decompress_array(compressed)))

        array[0:10] = np.ma.masked
        compressed = compress_array(array)
        self.assertTrue(np.all(array == decompress_array(compressed)))

    def test_compress_mapped_array(self):
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        values = 1 + np.arange(100) % 3
        array = MappedArray(values, values_mapping=mapping)
        compressed = compress_array(array)
        self.assertTrue(np.all(array == decompress_array(compressed)))

        array[0:10] = np.ma.masked
        compressed = compress_array(array)
        self.assertTrue(np.all(array == decompress_array(compressed)))


class TestPickle(unittest.TestCase):
    def test_pickle_mapped_array(self):
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        values = 1 + np.arange(100) % 3
        array = MappedArray(values, values_mapping=mapping)
        pickled = pickle.dumps(array)
        self.assertTrue(np.all(array == pickle.loads(pickled)))

        array[0:10] = np.ma.masked
        pickled = pickle.dumps(array)
        self.assertTrue(np.all(array == pickle.loads(pickled)))


class TestMappedArray(unittest.TestCase):
    mapping = {1: 'one', 2: 'two', 3: 'three'}

    def test_copy(self):
        values = [1, 2, 3, 2, 1, 2, 3, 2, 1]
        a = MappedArray(values, mask=[True] + [False] * 8, values_mapping=self.mapping)
        b = a.copy()
        self.assertFalse(a.raw is b.raw)
        self.assertTrue(np.all(a == b))
        self.assertEqual(a.values_mapping, b.values_mapping)

    def test_any_of(self):
        values = [1, 2, 3, 2, 1, 2, 3, 2, 1]
        a = MappedArray(values, mask=[True] + [False] * 8, values_mapping=self.mapping)
        result = a.any_of('one', 'three')
        self.assertEqual(
            result.tolist(),
            [None, False, True, False, True, False, True, False, True])
        self.assertRaises(ValueError, a.any_of, 'one', 'invalid')
        result = a.any_of('one', 'invalid', ignore_missing=True)
        self.assertEqual(
            result.tolist(),
            [None, False, False, False, True, False, False, False, True],
        )

    def test_create_from_list(self):
        values = [1, 2, 3]
        mask = [False, True, False]
        a = MappedArray(values, mask=mask, values_mapping=self.mapping)
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertEqual(a[2], 'three')

    def test_create_from_ma(self):
        values = [1, 2, 3]
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        self.assertEqual(a[2], 'three')

    def test_wrong_submasks(self):
        '''Fail setting a parameter with submasks not matching mask in array.'''
        values = [1, 2, 3, 3]
        mask = [False, True, False, True]
        sub1 = [False, True, True, True]
        submasks = {'sub1': sub1}
        array = np.ma.MaskedArray(values, mask)
        with self.assertRaises(ValueError):
            Parameter('Test', array=array, submasks=submasks)

    def test_get_slice(self):
        values = [1, 2, 3]
        mask = [False, True, False]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        a = a[:2]
        self.assertEqual(a[0], 'one')
        self.assertTrue(a[1] is np.ma.masked)
        # get mapped data value
        self.assertEqual(type(a), MappedArray)
        self.assertEqual(a.state['one'], [1])

    def test_set_slice(self):
        values = [1, 2, 3, 3]
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        a[:2] = ['two', 'three']  # this will unmask second item!
        self.assertEqual(a[0], 'two')
        self.assertEqual(list(a.raw), [2, 3, 3, np.ma.masked])
        self.assertEqual(a[1], 'three')  # updated value
        self.assertTrue(a[1] is not np.ma.masked)  # mask is lost
        self.assertTrue(a[3] is np.ma.masked)  # mask is maintained

        a.mask = False
        a[:3] = np.ma.masked
        self.assertEqual(list(a.raw.mask), [True, True, True, False])
        a[:] = np.ma.array([3, 3, 3, 3], mask=[False, False, True, True])
        self.assertEqual(list(a.raw.mask), [False, False, True, True])
        # set a slice to a single value
        a[2:] = 'one'
        self.assertEqual(list(a.raw[2:]), [1, 1])
        a[2:] = 3
        self.assertEqual(list(a.raw[2:]), [3, 3])
        # set a slice to a single element list (odd but consistent with numpy)
        a[2:] = [2]
        self.assertEqual(list(a.raw[2:]), [2, 2])
        a[2:] = ['three']
        self.assertEqual(list(a.raw[2:]), [3, 3])
        sub = [1, np.ma.masked]
        a[2:] = sub
        self.assertEqual(list(a.raw[2:]), [1, np.ma.masked])

    def test_set_slice_errors(self):
        values = [1, 2, 3, 3]
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        # value from assigned list is not in state
        sub = ['one', 'missing']
        with self.assertRaises(KeyError):
            a[2:] = sub
        # value from assigned list is not in values_mapping
        sub = [1, 100]
        with self.assertRaises(KeyError):
            a[2:] = sub
        # unequal number of arguments
        with self.assertRaises(ValueError):
            a[:] = ['one', 'one']

    def test_no_mapping(self):
        # values_mapping is no longer a requirement. (check no exception raised)
        self.assertTrue(all(MappedArray(np.arange(10)).data == np.arange(10)))

    def test_repr(self):
        values = [1, 2, 3, 3]
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        # ensure string vals is within repr
        self.assertTrue('one' in a.__repr__())

    def test_getitem_filters_boolean_array(self):
        "Tests __getitem__ and __eq__ and __ne__"
        ma = MappedArray(np.ma.arange(4, -1, step=-1), values_mapping={1: 'one', 2: 'two'})

        # boolean returned where: array == value
        #                                 4       3      2     >1<     0
        self.assertEqual(list(ma == 1), [False, False, False, True, False])
        self.assertEqual(list(ma != 1), [True, True, True, False, True])

        # Nice to Have : Overide == for state
        # boolean returned where: array == 'state'
        #                                     4       3     >2<      1     0
        self.assertEqual(list(ma == 'two'), [False, False, True, False, False])
        self.assertEqual(list(ma != 'two'), [True, True, False, True, True])

        n = np.arange(5)
        self.assertEqual(list(n[ma <= 1]), [3, 4])   # last two elements in ma are <= 1

        # boolean returned where: array == 'state'
        self.assertEqual(list(ma[ma <= 1]), ['one', '?'])  # last two elements in ma are <= 1

    def test_tolist(self):
        array = MappedArray([0] * 5 + [1] * 5, values_mapping={0: '-', 1: 'Warning'})
        self.assertEqual(array.tolist(), ['-'] * 5 + ['Warning'] * 5)
        array[2] = np.ma.masked
        self.assertEqual(array.tolist(), ['-'] * 2 + [None] + ['-'] * 2 + ['Warning'] * 5)

    def test_set_item(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        ma = MappedArray(np.ma.arange(1, 5), values_mapping=values_mapping)
        # Set single item.
        ma[0] = 'four'
        self.assertEqual(ma[0], 'four')
        ma[1] = 3
        self.assertEqual(ma[1], 'three')
        # Set multiple items with a list.
        ma[2:] = ['one', 'two']
        self.assertEqual(ma[2], 'one')
        self.assertEqual(ma[3], 'two')
        ma[2:] = [3, 4]
        self.assertEqual(ma[2], 'three')
        self.assertEqual(ma[3], 'four')
        # Set multiple items with a MaskedArray.
        ma[2:] = np.ma.MaskedArray([2, 3])
        self.assertEqual(ma[2], 'two')
        self.assertEqual(ma[3], 'three')
        ma[2:] = np.ma.MaskedArray([1.0, 2.0])
        self.assertEqual(ma[2], 'one')
        self.assertEqual(ma[3], 'two')
        ma[2:] = np.ma.MaskedArray([2, 3], mask=[True, False])
        self.assertTrue(ma[2] is np.ma.masked)
        self.assertEqual(ma[3], 'three')
        ma[np.array([True, False, True, True])] = np.ma.MaskedArray([4, 2, 3], mask=[False, False, True])
        self.assertEqual(ma[0], 'four')
        self.assertEqual(ma[2], 'two')
        self.assertTrue(ma[3] is np.ma.masked)
        # Set multiple items with a MappedArray.
        ma[:3] = MappedArray([1, 2, 3], values_mapping=values_mapping)
        self.assertEqual(ma[0], 'one')
        self.assertEqual(ma[1], 'two')
        self.assertEqual(ma[2], 'three')

    def test_array_equality(self):
        ma = MappedArray(np.ma.arange(1, 4), values_mapping={1: 'one', 2: 'two'})

        # unequal length arrays compared return False in np
        self.assertEqual(ma == ['one', 'two'], False)
        self.assertEqual(ma != ['one', 'two'], True)

        # mapped values
        np.testing.assert_array_equal(ma[:2] == ['one', 'two'], [True, True])
        np.testing.assert_array_equal(ma[:2] == ['one', 'one'], [True, False])
        np.testing.assert_array_equal(ma[:2] == ['INVALID', 'one'], [False, False])

        # where no mapping exists
        np.testing.assert_array_equal(ma == [1, 2, 3], [True, True, True])
        # test using dtype=int
        np.testing.assert_array_equal(ma == np.ma.array([1, 2, 3]), [True, True, True])
        # no mapping means you cannot find those values! Always get a FAIL
        # - sort out your values mapping!
        np.testing.assert_array_equal(ma == ['one', 'two', None], [True, True, False])
        np.testing.assert_array_equal(ma == ['one', 'two', '?'], [True, True, False])
        # test __ne__ (easy by comparison!)
        np.testing.assert_array_equal(ma != ['one', 'two', '?'], [False, False, True])

        # masked values
        ma[0] = np.ma.masked
        np.testing.assert_array_equal(ma[:2] == [np.ma.masked, 2], [True, True])
        # can't compare lists with numpy arrays
        np.testing.assert_array_equal(ma[:2] == [np.ma.masked, 'two'], [True, True])

    def test_mapped_arrays_equality(self):
        # Same arrays using values_mapping (underlying buffer is different)
        ma = MappedArray(np.ma.arange(1, 4), values_mapping={1: 'one', 2: 'two', 3: 'three'})
        ma2 = MappedArray([11, 12, 13], values_mapping={11: 'one', 12: 'two', 13: 'three'})
        np.testing.assert_array_equal(ma == ma2, [True, True, True])

        # One value not in values_mapping
        ma2 = MappedArray([11, 12, 14], values_mapping={11: 'one', 12: 'two', 13: 'three'})
        np.testing.assert_array_equal(ma == ma2, [True, True, False])

        # Values not in values_mapping are compared among themselves
        ma = MappedArray([1, 2, 3, 3], values_mapping={1: 'one', 2: 'two'})
        ma2 = MappedArray([11, 12, 14, 13], values_mapping={11: 'one', 12: 'two'})
        np.testing.assert_array_equal(ma == ma2, [True, True, False, False])

        ma = MappedArray([1, 2, 3, 13], values_mapping={1: 'one', 2: 'two'})
        ma2 = MappedArray([11, 12, 14, 13], values_mapping={11: 'one', 12: 'two'})
        np.testing.assert_array_equal(ma == ma2, [True, True, False, True])

        # One value missing from values_mapping in one mapped array
        ma = MappedArray([1, 2, 3, 13], values_mapping={1: 'one', 2: 'two', 13: 'three'})
        ma2 = MappedArray([11, 12, 14, 13], values_mapping={11: 'one', 12: 'two'})
        np.testing.assert_array_equal(ma == ma2, [True, True, False, False])

        # Masked values are ignored (like numpy masked arrays)
        ma = MappedArray([1, 2, 3, 13], values_mapping={1: 'one', 2: 'two'})
        ma[1] = np.ma.masked
        ma2 = MappedArray([11, 12, 14, 13], values_mapping={11: 'one', 12: 'two'})
        ma2[2] = np.ma.masked
        np.testing.assert_array_equal(
            ma == ma2,
            np.ma.array([True, True, False, True], mask=[False, True, True, False])
        )

    def test_array_inequality_type_and_mask(self):
        data = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]

        array = np.ma.masked_array(data=data, mask=False)
        array = MappedArray(array, values_mapping={0: 'Off', 1: 'On'})

        expected = np.ma.array([not bool(x) for x in data])

        np.testing.assert_array_equal(array != 'On', expected)

        # Ensure that __ne__ is returning a boolean array!
        np.testing.assert_array_equal(
            str(array != 'On'),
            '[True True True True False False True True False True]')

        array[array != 'On'] = np.ma.masked
        np.testing.assert_array_equal(array.mask, expected)

    def test_array_finalize(self):
        """
        Numpy in some cases creates an array derived from the arguments instead
        of modifying the original object in place.

        In those cases new_array.__array_finalize__() is called to apply all
        specific initialisations.

        In case of MappedArray it should copy values_mapping from the master
        object.
        """
        data = [0, 0, 0, 0, 1, 1, 0, 0, 1, 0]

        array = np.ma.masked_array(data=data, mask=False)
        array = MappedArray(array, values_mapping={0: 'Off', 1: 'On'})
        # if the __array_finalize__ wasn't called this would raise exception:
        # AttributeError: 'MappedArray' object has no attribute 'values_mapping'
        result = np.ma.masked_less(array, 1.0)
        self.assertEqual(array.values_mapping, result.values_mapping)

    def test_duplicate_values(self):
        values_mapping = {0: 'A', 1: 'A', 2: 'B', 3: 'C', 5: 'C'}
        data = np.ma.masked_array([0, 1, 2, 3, 4, 5], mask=False)
        array = MappedArray(data, values_mapping=values_mapping)
        self.assertEqual(array[0], 'A')
        self.assertEqual(array[1], 'A')
        self.assertEqual(array[2], 'B')
        self.assertEqual(array[3], 'C')
        self.assertEqual(array[4], '?')
        self.assertEqual(array[5], 'C')

        self.assertEqual(array.state['A'], [0, 1])
        self.assertEqual(array.state['B'], [2])
        self.assertEqual(array.state['C'], [3, 5])

        self.assertEqual((array == 'A').tolist(), [True, True, False, False, False, False])

    def test_missing_state(self):
        """Ensure the array compared to incorrect state raises KeyError."""
        values_mapping = {0: 'A', 1: 'B', 2: 'C'}
        array = MappedArray(
            [0, 0, 0, 1, 2, 1, 0, 0], mask=[True] * 2 + [False] * 5 + [True], values_mapping=values_mapping)
        self.assertEqual((array == 'A').tolist(), [None, None, True, False, False, False, True, None])
        self.assertRaises(KeyError, array.__eq__, 'D')
        self.assertRaises(KeyError, array.__ne__, 'E')
        self.assertRaises(KeyError, array.__gt__, 'F')
        self.assertRaises(KeyError, array.__ge__, 'G')
        self.assertRaises(KeyError, array.__lt__, 'H')
        self.assertRaises(KeyError, array.__le__, 'I')

    def test_missing_value(self):
        """Ensure the array compared to incorrect numeric value returns False."""
        values_mapping = {0: 'A', 1: 'B', 2: 'C'}
        array = MappedArray(
            [0, 0, 0, 1, 2, 1, 0, 0], mask=[True] * 2 + [False] * 5 + [True], values_mapping=values_mapping)
        self.assertFalse(np.any(array == 5))

    def test_get_state_value(self):
        values_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
        data = np.ma.masked_array([1, 2, 3, 4, 5], mask=False)
        array = MappedArray(data, values_mapping=values_mapping)
        for raw, state in values_mapping.items():
            self.assertEqual(array.get_state_value(state), [raw])

    def test_view_from_ndarray(self):
        """
        Making a MappedArray view from a ndarray should result in a
        MappedArray with an empty values_mapping.
        """
        array = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        mapped_array = array.view(MappedArray)
        np.testing.assert_array_equal(mapped_array.raw, array)
        np.testing.assert_array_equal(mapped_array.mask, [False] * len(array))
        self.assertDictEqual(mapped_array.values_mapping, {})


class TestParameterSubmasks(unittest.TestCase):
    """Test ParameterSubmasks proxy."""
    def test_delitem(self):
        ps = ParameterSubmasks()
        ps['test'] = np.zeros(100)
        del ps['test']
        self.assertNotIn('test', ps)

    def test_compression(self):
        ps = ParameterSubmasks(compress=True)
        mask = np.zeros(100)
        ps['test'] = mask
        # implicit conversion to bool Numpy array
        self.assertEqual(ps['test'].dtype, np.bool)
        self.assertTrue(np.all(mask == ps['test']))


if __name__ == '__main__':
    unittest.main()
