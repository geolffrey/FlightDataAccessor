from __future__ import print_function

import copy
import unittest

import numpy as np

from flightdataaccessor.datatypes.parameter import MappedArray, Parameter


class TestMappedArray(unittest.TestCase):
    mapping = {1: 'one', 2: 'two', 3: 'three'}

    def test_any_of(self):
        values = [1, 2, 3, 2, 1, 2, 3, 2, 1]
        a = MappedArray(values, mask=[True] + [False] * 8,
                        values_mapping=self.mapping)
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
        """Fail setting a parameter with submasks not matching mask in array."""
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
        self.assertEqual(list(a.raw.data[2:]), [1, 1])
        a[2:] = 3
        self.assertEqual(list(a.raw.data[2:]), [3, 3])
        # set a slice to a single element list (odd but consistent with numpy)
        a[2:] = [2]
        self.assertEqual(list(a.raw.data[2:]), [2, 2])
        a[2:] = ['three']
        self.assertEqual(list(a.raw.data[2:]), [3, 3])
        # unequal number of arguments
        self.assertRaises(ValueError, a.__setitem__, slice(-3, None), ['one', 'one'])

    def test_no_mapping(self):
        # values_mapping is no longer a requirement. (check no exception raised)
        self.assertTrue(all(MappedArray(np.arange(10)).data == np.arange(10)))

    def test_repr(self):
        values = [1, 2, 3, 3]
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=self.mapping)
        # ensure string vals is within repr
        print(a.__repr__())
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

        # check __repr__ and __str__ work
        self.assertEqual((ma == 'two').__str__(), '[False False  True False False]')
        self.assertEqual((ma == 'two').__repr__(), '''\
masked_array(data = [False False  True False False],
             mask = False,
       fill_value = True)
''')
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
        self.assertEquals(array.values_mapping, result.values_mapping)

    def test_duplicate_values(self):
        values_mapping = {0: 'A', 1: 'A', 2: 'B', 3: 'C', 5: 'C'}
        data = [0, 1, 2, 3, 4, 5]

        array = np.ma.masked_array(data, mask=False)
        array = MappedArray(array, values_mapping=values_mapping)
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
        values_mapping = {0: 'A', 1: 'B', 2: 'C'}
        array = MappedArray([0, 0, 0, 1, 2, 1, 0, 0], mask=[True] * 2 + [False] * 5 + [True], values_mapping=values_mapping)
        self.assertEqual((array == 'A').tolist(), [None, None, True, False, False, False, True, None])
        self.assertRaises(KeyError, array.__eq__, 'D')
        self.assertRaises(KeyError, array.__ne__, 'E')
        self.assertRaises(KeyError, array.__gt__, 'F')
        self.assertRaises(KeyError, array.__ge__, 'G')
        self.assertRaises(KeyError, array.__lt__, 'H')
        self.assertRaises(KeyError, array.__le__, 'I')


class TestParameter(unittest.TestCase):
    def get_parameter(self, frequency=1, with_mask=True):
        array = np.ma.arange(100)
        mask = np.zeros(100, dtype=np.bool)
        if with_mask:
            mask[:3] = [1, 1, 0]
            array.mask = mask
            mask1 = np.ma.zeros(100, dtype=np.bool)
            mask1[:3] = [1, 0, 0]
            mask2 = np.ma.zeros(100, dtype=np.bool)
            mask2[:3] = [1, 1, 0]
            submasks = {'mask1': mask1, 'mask2': mask2}
        else:
            submasks = {}
        return Parameter('Test', array=array, submasks=submasks, frequency=frequency)

    def test_parameter(self):
        p_name = 'param'
        p = Parameter(p_name)
        self.assertEqual(p.name, p_name)
        self.assertEqual(p.array.size, 0)
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

    def test_multivalue_parameter_float_values(self):
        values = [17.5, 10.5, 9]
        mask = [False, True, False]
        array = np.ma.MaskedArray(values, mask)
        mapping = {'17.5': 'One', 10.5: 'Two', 5: 'Three'}
        p = Parameter('param', array=array, values_mapping=mapping)
        self.assertEqual(p.array[0], 'One')
        self.assertEqual(p.raw_array[0], 17.5)
        self.assertTrue(p.array[1] is np.ma.masked)
        self.assertTrue(p.raw_array[1] is np.ma.masked)
        # Get a value not in the mapping
        self.assertEqual(p.array[2], '?')
        self.assertEqual(p.raw_array[2], 9)

    def test_combine_submasks(self):
        array = np.array([10, 20, 30])
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        })
        self.assertEqual(p.combine_submasks().tolist(), [1, 1, 0])

    def test_get_array(self):
        array = np.ma.array([10, 20, 30], mask=[1, 1, 0])
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        })
        self.assertEqual(p.get_array().tolist(), [None, None, 30])
        self.assertEqual(p.get_array('mask1').tolist(), [None, 20, 30])
        self.assertEqual(p.get_array('mask2').tolist(), [None, None, 30])

    def test_get_array__mapped(self):
        array = np.ma.array([1, 2, 3], mask=[1, 1, 0])
        values_mapping = {1: 'One', 2: 'Two', 3: 'Three'}
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        }, values_mapping=values_mapping)
        self.assertEqual(p.get_array().raw.tolist(), [None, None, 3])
        self.assertEqual(p.get_array('mask1').raw.tolist(), [None, 2, 3])
        self.assertEqual(p.get_array('mask2').raw.tolist(), [None, None, 3])
        self.assertTrue(isinstance(p.get_array('mask1'), MappedArray))

    def test_validate_mask(self):
        """Mask validation."""
        p = self.get_parameter()
        # validate the parameter itself
        self.assertTrue(p.validate_mask())
        # validate arbitrary array and submasks
        self.assertTrue(p.validate_mask(array=p.array, submasks=p.submasks))
        # validate arbitrary array without a mask
        self.assertTrue(p.validate_mask(array=p.array.data))

        # fail validation of array with masked items if no corresponding submasks is passed
        # XXX: for backwards compatibility raising of this exception is disabled in normal mode
        # TODO: implement strict mode and make it the default after refactoring is complete
        # with self.assertRaises(ValueError):
        #     self.assertTrue(p.validate_mask(array=p.array))

        # fail validation of array with masked items if submasks passed have different sizes
        with self.assertRaises(ValueError):
            self.assertTrue(p.validate_mask(array=p.array[:10], submasks=p.submasks))

        # fail validation of array with masked items if submasks passed have different keys
        with self.assertRaises(ValueError):
            submasks = copy.copy(p.submasks)
            submasks = {'mask1': submasks['mask1']}
            self.assertTrue(p.validate_mask(array=p.array, submasks=submasks))

        # fail validation if combined submasks are not equivalent to array.mask
        with self.assertRaises(ValueError):
            submasks = copy.copy(p.submasks)
            array = p.array.copy()
            array.mask[0] = False
            self.assertTrue(p.validate_mask(array=array, submasks=p.submasks))

    def test_slice(self):
        """Slice parameter

        Number of samples independent from sample rate."""
        p = self.get_parameter()
        p2 = p.slice(slice(10, 20))
        self.assertEquals(p2.array.size, 10)
        self.assertEquals(p2.submasks['mask1'].size, 10)
        self.assertEquals(p2.submasks['mask2'].size, 10)
        # 2Hz
        p = self.get_parameter(frequency=2)
        p2 = p.slice(slice(10, 20))
        self.assertEquals(p2.array.size, 10)
        self.assertEquals(p2.submasks['mask1'].size, 10)
        self.assertEquals(p2.submasks['mask2'].size, 10)

    def test_trim(self):
        """Trim parameter

        Number of samples dependent on sample rate."""
        p = self.get_parameter(frequency=.5)
        p2 = p.trim(start_offset=10, stop_offset=20)
        self.assertEquals(p2.array.size, 5)
        self.assertEquals(p2.submasks['mask1'].size, 5)
        self.assertEquals(p2.submasks['mask2'].size, 5)

        p = self.get_parameter(frequency=2)
        p2 = p.trim(start_offset=10, stop_offset=20)
        self.assertEquals(p2.array.size, 20)
        self.assertEquals(p2.submasks['mask1'].size, 20)
        self.assertEquals(p2.submasks['mask2'].size, 20)

    def test_trim_superframe_boundary(self):
        """Trim parameter to a window in seconds.

        Number of samples dependent on sample rate."""
        p = self.get_parameter(frequency=.5, with_mask=False)
        p2 = p.trim(start_offset=10, stop_offset=20, superframe_boundary=True)
        # the window is implicitely extended to superframe boundaries, which is 64 seconds wide, in this case
        # start_offset=0, stop_offset=64
        self.assertEquals(p2.array.size, 32)
        # unrequested edges are masked
        self.assertTrue(np.all(p2.array.mask[:5]))
        self.assertTrue(np.all(p2.array.mask[10:]))
        # requested data is not masked
        self.assertFalse(np.any(p2.array.mask[5:10]))

    def test_trim_superframe_boundary_padding(self):
        p = self.get_parameter(frequency=2, with_mask=False)
        p2 = p.trim(start_offset=10, stop_offset=20, superframe_boundary=True)
        # because the array size is 100 which is only 50 seconds, the whole data is returned and padded at the end to
        # complete superframes
        # result data has size greater than the original due to superframe padding
        self.assertEquals(p2.array.size, 128)
        # padding submask is added
        self.assertTrue('padding' in p2.submasks)
        # unrequested edges are masked
        self.assertTrue(np.all(p2.array.mask[:20]))
        self.assertTrue(np.all(p2.array.mask[40:]))
        # requested data is not masked
        self.assertFalse(np.any(p2.array.mask[20:40]))

    def test_trim_superframe_boundary_no_padding(self):
        p = self.get_parameter(frequency=2, with_mask=False)
        p2 = p.trim(start_offset=10, stop_offset=20, pad=False, superframe_boundary=True)
        # because the array size is 100 which is only 50 seconds, the whole data is returned
        self.assertEquals(p2.array.size, 100)
        # unrequested edges are masked
        self.assertTrue(np.all(p2.array.mask[:20]))
        self.assertTrue(np.all(p2.array.mask[40:]))
        # requested data is not masked
        self.assertFalse(np.any(p2.array.mask[20:40]))

    def test_extend(self):
        """Extend parameter without a mask."""
        p = self.get_parameter()
        p.extend(p.array.data)
        self.assertEquals(p.array.size, 200)
        self.assertEquals(p.submasks['mask1'].size, 200)
        self.assertEquals(p.submasks['mask2'].size, 200)

    def test_extend_multistate_int(self):
        """Extend multistate parameter with a list of integers."""
        array = np.ma.array([1, 2, 3], mask=[1, 1, 0])
        values_mapping = {1: 'One', 2: 'Two', 3: 'Three'}
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        }, values_mapping=values_mapping)
        p.extend([1, 2, 3])
        np.testing.assert_array_equal([None, None, 'Three', 'One', 'Two', 'Three'], p.array)

    def test_extend_multistate_str(self):
        """Extend multistate parameter with a list of valid strings."""
        array = np.ma.array([1, 2, 3], mask=[1, 1, 0])
        values_mapping = {1: 'One', 2: 'Two', 3: 'Three'}
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        }, values_mapping=values_mapping)
        p.extend(['One', 'Two', 'Three'])
        np.testing.assert_array_equal([None, None, 'Three', 'One', 'Two', 'Three'], p.array)

    def test_extend_multistate_mapped(self):
        """Extend multistate parameter with a MappedArray."""
        array = np.ma.array([1, 2, 3], mask=[1, 1, 0])
        values_mapping = {1: 'One', 2: 'Two', 3: 'Three'}
        p = Parameter('Submasks', array=array, submasks={
            'mask1': np.array([1, 0, 0], dtype=np.bool8),
            'mask2': np.array([1, 1, 0], dtype=np.bool8),
        }, values_mapping=values_mapping)
        p.extend(MappedArray([1, 2, 3], values_mapping=values_mapping))
        np.testing.assert_array_equal([None, None, 'Three', 'One', 'Two', 'Three'], p.array)

    def test_extend_with_mask(self):
        """Extend parameter array with a mask."""
        p = self.get_parameter()
        p.extend(p.array, submasks=p.submasks)
        self.assertEquals(p.array.size, 200)
        self.assertEquals(p.submasks['mask1'].size, 200)
        self.assertEquals(p.submasks['mask2'].size, 200)

    def test_extend_parameter(self):
        """Extend parameter with another parameter."""
        p = self.get_parameter()
        p.extend(p)
        self.assertEquals(p.array.size, 200)
        self.assertEquals(p.submasks['mask1'].size, 200)
        self.assertEquals(p.submasks['mask2'].size, 200)

        with self.assertRaises(ValueError):
            # submasks argument is invalid when extending with a Parameter
            p.extend(p, submasks=p.submasks)

    def test_extend_failures(self):
        """Test failures to extend parameter array."""
        p = self.get_parameter()
        array = p.array.copy()
        # break the mask: the combined submasks give different mask
        array.mask[0] = False
        with self.assertRaises(ValueError):
            p.extend(array, submasks=p.submasks)

    def test_is_compatible(self):
        """Test compatibility checks between parameters."""
        p1 = self.get_parameter()
        p2 = self.get_parameter()
        self.assertTrue(p1.is_compatible(p2))
        self.assertTrue(p2.is_compatible(p1))

        p2.name = p2.name + 'X'
        self.assertFalse(p1.is_compatible(p2))
        self.assertFalse(p2.is_compatible(p1))

        p2 = self.get_parameter(frequency=2)
        self.assertFalse(p1.is_compatible(p2))
        self.assertFalse(p2.is_compatible(p1))

        p2 = self.get_parameter()
        p2.unit = 'unknown'
        self.assertFalse(p1.is_compatible(p2))
        self.assertFalse(p2.is_compatible(p1))

        p2 = self.get_parameter()
        p2.offset += 0.25
        self.assertFalse(p1.is_compatible(p2))
        self.assertFalse(p2.is_compatible(p1))

    def test_update_submask(self):
        """Update a submask and ensure the mask is in sync."""
        p = self.get_parameter()
        old_mask = p.submasks['mask1'].copy()
        mask = old_mask.copy()
        mask[3] = True
        p.update_submask('mask1', mask)
        self.assertFalse(np.all(mask == old_mask))
        self.assertTrue(p.validate_mask())


if __name__ == '__main__':
    unittest.main()
