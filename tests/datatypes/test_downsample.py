import unittest

import numpy as np

from flightdataaccessor.datatypes import downsample, parameter


class MostCommonValueTest(unittest.TestCase):
    def test_most_common_value_list(self):
        array = [1, 2, 3, 4, 5, 3, 2, 3]
        self.assertEquals(downsample.most_common_value(array), 3)

    def test_most_common_value_array(self):
        array = np.array([1, 2, 3, 4, 5, 3, 2, 3])
        self.assertEquals(downsample.most_common_value(array), 3)

    def test_most_common_value_masked_array(self):
        array = np.ma.array([1, 2, 3, 4, 5, 3, 2, 3], mask=[False] * 8)
        array.mask[2] = True
        array.mask[-1] = True
        self.assertEquals(downsample.most_common_value(array), 2)

    def test_most_common_value_list_of_strings(self):
        array = ['1', '2', '3', '4', '5', '3', '2', '3']
        self.assertEquals(downsample.most_common_value(array), '3')

    def test_most_common_value_array_of_strings(self):
        array = np.array(['1', '2', '3', '4', '5', '3', '2', '3'])
        self.assertEquals(downsample.most_common_value(array), '3')

    def test_most_common_value_mapped_array(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        array = parameter.MappedArray([1, 2, 3, 4, 4, 3, 2, 3], values_mapping=values_mapping)
        self.assertEquals(downsample.most_common_value(array), 3)

    def test_most_common_value_high_threshold(self):
        array = [1, 2, 3, 4, 5, 3, 2, 3]
        self.assertEquals(downsample.most_common_value(array, 10), None)


class DownsampleTest(unittest.TestCase):
    def test_downsample_list(self):
        array = list(range(100))
        downsampled = downsample.downsample(array, 4)
        # 2 buckets, 2 samples per bucket, min and max
        np.testing.assert_array_equal(downsampled, [0, 49, 50, 99])
        array = array[::-1]
        downsampled = downsample.downsample(array, 4)
        np.testing.assert_array_equal(downsampled, [99, 50, 49, 0])

    def test_downsample_array(self):
        array = np.arange(100)
        downsampled = downsample.downsample(array, 4)
        # 2 buckets, 2 samples per bucket, min and max
        np.testing.assert_array_equal(downsampled, [0, 49, 50, 99])
        array = array[::-1]
        downsampled = downsample.downsample(array, 4)
        np.testing.assert_array_equal(downsampled, [99, 50, 49, 0])

    def test_downsample_array_masked(self):
        mask = np.zeros(100, dtype=np.bool)
        mask[:50] = True
        array = np.ma.arange(100)
        array.mask = mask
        downsampled = downsample.downsample(array, 4)
        # 2 buckets, 2 samples per bucket, min and max
        np.testing.assert_array_equal(downsampled, [np.ma.masked, np.ma.masked, 50, 99])
        array = array[::-1]
        downsampled = downsample.downsample(array, 4)
        np.testing.assert_array_equal(downsampled, [99, 50, np.ma.masked, np.ma.masked])

    def test_downsample_list_of_strings(self):
        array = ['one', 'two', 'three', 'four', 'one'] * 20
        downsampled1 = downsample.downsample(array, 4)
        downsampled2 = downsample.downsample_most_common_value(array, 4)
        # implicit use of the nonnumeric algorithm
        np.testing.assert_array_equal(downsampled1, downsampled2)
        np.testing.assert_array_equal(downsampled1, ['one', 'one', 'one', 'one'])

    def test_downsample_array_of_strings(self):
        array = np.array(['one', 'two', 'three', 'four', 'one'] * 20)
        downsampled1 = downsample.downsample(array, 4)
        downsampled2 = downsample.downsample_most_common_value(array, 4)
        # implicit use of the nonnumeric algorithm
        np.testing.assert_array_equal(downsampled1, downsampled2)
        np.testing.assert_array_equal(downsampled1, ['one', 'one', 'one', 'one'])

    def test_downsample_array_of_strings_masked(self):
        mask = np.zeros(100, dtype=np.bool)
        mask[:50] = True
        array = np.ma.array(['one', 'two', 'three', 'four', 'one'] * 20)
        array.mask = mask
        downsampled1 = downsample.downsample(array, 4)
        downsampled2 = downsample.downsample_most_common_value(array, 4)
        # implicit use of the nonnumeric algorithm
        np.testing.assert_array_equal(downsampled1, downsampled2)
        np.testing.assert_array_equal(downsampled1, [np.ma.masked, np.ma.masked, 'one', 'one'])

    def test_downsample_mapped_array(self):
        values_mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        array = parameter.MappedArray([1, 2, 3, 4, 1] * 20, values_mapping=values_mapping)
        downsampled1 = downsample.downsample(array, 4)
        downsampled2 = downsample.downsample_most_common_value(array, 4)
        # implicit use of the nonnumeric algorithm
        self.assertTrue(np.all(downsampled1 == downsampled2))
        np.testing.assert_array_equal(downsampled1, ['one', 'one', 'one', 'one'])

    def test_downsample_mapped_array_masked(self):
        mask = np.zeros(100, dtype=np.bool)
        mask[:50] = True
        values_mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        array = parameter.MappedArray([1, 2, 3, 4, 1] * 20, mask=mask, values_mapping=values_mapping)
        downsampled1 = downsample.downsample(array, 4)
        downsampled2 = downsample.downsample_most_common_value(array, 4)
        # implicit use of the nonnumeric algorithm
        self.assertTrue(np.all(downsampled1 == downsampled2))
        np.testing.assert_array_equal(downsampled1, [np.ma.masked, np.ma.masked, 'one', 'one'])

    def test_downsample_mapped_array_remainder(self):
        mask = np.zeros(100, dtype=np.bool)
        mask[:50] = True
        values_mapping = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
        array = parameter.MappedArray([1, 2, 3, 4, 1] * 20, mask=mask, values_mapping=values_mapping)
        # uneven split of samples (34, 34 and 32 samples)
        downsampled = downsample.downsample(array, 3)
        np.testing.assert_array_equal(downsampled, [np.ma.masked, 'one', 'one'])
