from __future__ import print_function

import copy
import unittest

import numpy as np

from flightdataaccessor.datatypes.parameter import MappedArray, Parameter


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
        p2 = p.trim(start_offset=10, stop_offset=20, pad_subframes=False)
        self.assertEquals(p2.array.size, 5)
        self.assertEquals(p2.submasks['mask1'].size, 5)
        self.assertEquals(p2.submasks['mask2'].size, 5)

        p = self.get_parameter(frequency=2)
        p2 = p.trim(start_offset=10, stop_offset=20, pad_subframes=False)
        self.assertEquals(p2.array.size, 20)
        self.assertEquals(p2.submasks['mask1'].size, 20)
        self.assertEquals(p2.submasks['mask2'].size, 20)

    def test_trim_pad_subframes(self):
        """Trim parameter to a window in seconds.

        Number of samples dependent on sample rate."""
        p = self.get_parameter(frequency=.5, with_mask=False)
        p2 = p.trim(start_offset=10, stop_offset=20, pad_subframes=64)
        # the window is implicitely extended to superframe boundaries, which is 64 seconds wide, in this case
        # start_offset=0, stop_offset=64
        self.assertEquals(p2.array.size, 32)
        # unrequested edges are masked
        self.assertTrue(np.all(p2.array.mask[:5]))
        self.assertTrue(np.all(p2.array.mask[10:]))
        # requested data is not masked
        self.assertFalse(np.any(p2.array.mask[5:10]))

    def test_trim_pad_subframes_padding(self):
        p = self.get_parameter(frequency=2, with_mask=False)
        p2 = p.trim(start_offset=10, stop_offset=20, pad_subframes=64)
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
