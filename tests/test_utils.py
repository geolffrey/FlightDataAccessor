import errno
import os
import unittest

import h5py
import numpy as np

import flightdataaccessor as fda
from flightdataaccessor.utils import concat_hdf, strip_hdf, write_segment

TEST_DATA_DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
TEMP_DIR_PATH = os.path.join(TEST_DATA_DIR_PATH, 'temp')
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')


class CreateHDFForTest(object):
    '''
    Test classes can derive from this class to be able to create an HDF file
    for testing.
    '''
    def _create_hdf_test_file(self, hdf_path):
        self.data_secs = 128
        with h5py.File(hdf_path, 'w') as hdf_file:
            hdf_file.attrs['duration'] = self.data_secs
            series = hdf_file.create_group('series')
            # 'IVV' - 1Hz parameter.
            ivv_group = series.create_group('IVV')
            self.ivv_frequency = 1
            ivv_group.attrs['frequency'] = self.ivv_frequency
            self.ivv_supf_offset = 2.1
            ivv_group.attrs['offset'] = self.ivv_supf_offset
            self.ivv_data = np.arange(self.data_secs * self.ivv_frequency,
                                      dtype=np.float)
            self.ivv_mask = np.array([False] * len(self.ivv_data))
            ivv_group.create_dataset('data', data=self.ivv_data)
            ivv_group.create_dataset('mask', data=self.ivv_mask)
            # 'WOW' - 4Hz parameter.
            wow_group = series.create_group('WOW')
            self.wow_frequency = 4
            wow_group.attrs['frequency'] = self.wow_frequency
            self.wow_data = np.arange(self.data_secs * self.wow_frequency,
                                      dtype=np.float)
            self.wow_mask = np.array([False] * len(self.wow_data))
            wow_group.create_dataset('data', data=self.wow_data)
            wow_group.create_dataset('mask', data=self.wow_mask)
            # 'DME' - 0.25Hz parameter.
            dme_group = series.create_group('DME')
            self.dme_frequency = 0.25
            dme_group.attrs['frequency'] = self.dme_frequency
            self.dme_data = np.arange(self.data_secs * self.dme_frequency,
                                      dtype=np.float)
            self.dme_mask = np.array([False] * len(self.dme_data))
            dme_group.create_dataset('data', data=self.dme_data)
            dme_group.create_dataset('mask', data=self.dme_mask)


class TestConcatHDF(unittest.TestCase):
    def setUp(self):
        try:
            os.makedirs(TEMP_DIR_PATH)
        except OSError as err:
            if err.errno == errno.EEXIST:
                pass

        self.hdf_path_1 = os.path.join(TEMP_DIR_PATH,
                                       'concat_hdf_1.hdf5')
        self.hdf_data_1 = np.arange(160, dtype=np.float)
        self.hdf_path_2 = os.path.join(TEMP_DIR_PATH,
                                       'concat_hdf_2.hdf5')
        self.hdf_data_2 = np.arange(240, dtype=np.float)

        # Create test hdf files.
        with fda.open(self.hdf_path_1, mode='w') as fdf:
            fdf.frame_type = '737-3C'
            fdf['PARAM'] = fda.Parameter('PARAM', array=self.hdf_data_1, frequency=8)

        with fda.open(self.hdf_path_2, mode='w') as fdf:
            fdf.frame_type = '737-3C'
            fdf['PARAM'] = fda.Parameter('PARAM', array=self.hdf_data_2, frequency=8)

        self.hdf_out_path = os.path.join(TEMP_DIR_PATH, 'concat_out.dat')

    def test_concat_hdf__without_dest(self):
        self.__test_concat_hdf()

    def test_concat_hdf__with_dest(self):
        self.__test_concat_hdf(dest=self.hdf_out_path)

    def __test_concat_hdf(self, dest=None):
        '''
        Tests that the dataset within the path matching
        'series/<Param Name>/data' is concatenated, while other datasets and
        attributes are unaffected.
        '''
        with self.assertWarns(DeprecationWarning):
            out_path = concat_hdf((self.hdf_path_1, self.hdf_path_2), dest=dest)
        if dest:
            self.assertEqual(dest, out_path)
        with h5py.File(out_path, 'r') as hdf_out_file:
            series = hdf_out_file
            param = series['PARAM']
            self.assertEqual(param.attrs['frequency'], 8)
            data_result = param['data'][:]
            data_expected_result = np.concatenate((self.hdf_data_1, self.hdf_data_2))
            np.testing.assert_array_equal(data_result, data_expected_result)
            # self.assertEqual(hdf_out_file.attrs['frame_type'], '737-3C')
            self.assertEqual(hdf_out_file.attrs['duration'], 50)

    def tearDown(self):
        for file_path in (self.hdf_path_1, self.hdf_path_2, self.hdf_out_path):
            try:
                os.remove(file_path)
            except OSError as err:
                if err.errno != errno.ENOENT:
                    raise


class TestStripHDF(unittest.TestCase, CreateHDFForTest):
    def setUp(self):
        self.hdf_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_for_split_hdf.hdf5')
        self._create_hdf_test_file(self.hdf_path)
        self.out_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_split.hdf5')

    def tearDown(self):
        os.unlink(self.out_path)

    def test_strip_hdf_all(self):
        '''
        Do not keep any parameters.
        '''
        with self.assertWarns(DeprecationWarning):
            strip_hdf(self.hdf_path, [], self.out_path)
        with h5py.File(self.out_path, 'r') as hdf_file:
            series = hdf_file
            self.assertEqual(list(series.keys()), [])

    def test_strip_hdf_ivv(self):
        params_to_keep = ['IVV']
        with self.assertWarns(DeprecationWarning):
            strip_hdf(self.hdf_path, params_to_keep, self.out_path)
        with h5py.File(self.out_path, 'r') as hdf_file:
            series = hdf_file
            self.assertEqual(list(series.keys()), params_to_keep)
            # Ensure datasets are unchanged.
            self.assertTrue(all(series['IVV']['data'][:] == self.ivv_data))
            # self.assertTrue(all(series['series']['IVV']['mask'][:] == self.ivv_mask))
            # Ensure attributes are unchanged.
            self.assertEqual(series['IVV'].attrs['offset'],
                             self.ivv_supf_offset)
            self.assertEqual(series['IVV'].attrs['frequency'],
                             self.ivv_frequency)

    def test_strip_hdf_dme_wow(self):
        '''
        Does not test that datasets and attributes are maintained, see
        test_strip_hdf_ivv.
        '''
        params_to_keep = ['DME', 'WOW']
        with self.assertWarns(DeprecationWarning):
            strip_hdf(self.hdf_path, params_to_keep, self.out_path)
        with h5py.File(self.out_path, 'r') as hdf_file:
            series = hdf_file
            self.assertEqual(list(series.keys()), params_to_keep)


class TestWriteSegment(unittest.TestCase, CreateHDFForTest):
    def setUp(self):
        self.hdf_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_for_write_segment.hdf5')
        self._create_hdf_test_file(self.hdf_path)
        self.out_path = os.path.join(TEMP_DIR_PATH,
                                     'hdf_segment.hdf5')

    def assertItemsEqual(self, l1, l2):
        self.assertEqual(list(l1), list(l2))

    @unittest.skip('Rewrite this test to use the API to prepare the test data and to verify it')
    def test_write_segment__start_and_stop(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(10, 17)
        dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=4)
        self.assertEqual(dest, self.out_path)

        frame_boundary_segment = slice(8, 20)

        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['offset'],
                             self.ivv_supf_offset)
            # mask = merge_masks(ivv_group['submasks'][:])
            mask = False
            ivv_result = np.ma.masked_array(ivv_group['data'][:], mask=mask)
            ivv_expected_result = np.arange(
                frame_boundary_segment.start * self.ivv_frequency,
                frame_boundary_segment.stop * self.ivv_frequency,
                dtype=np.float)
            ivv_expected_result = np.ma.masked_array(
                ivv_expected_result, mask=[True] * 2 + [False] * 7 + [True] * 3)
            self.assertEqual(ivv_result.tolist(), ivv_expected_result.tolist())
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(
                frame_boundary_segment.start * self.wow_frequency,
                frame_boundary_segment.stop * self.wow_frequency,
                dtype=np.float)
            self.assertEqual(list(wow_result), list(wow_expected_result))
            # 'DME' - 0.25Hz parameter.
            dme_group = hdf_file['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]  # array([ 3.,  4.])
            dme_expected_result = np.arange(2, 5, dtype=np.float)
            self.assertEqual(list(dme_result), list(dme_expected_result))
            self.assertEqual(
                hdf_file.attrs['duration'],
                frame_boundary_segment.stop - frame_boundary_segment.start)

        # Write segment on superframe boundary.
        dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=64)
        self.assertEqual(dest, self.out_path)

        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['offset'],
                             self.ivv_supf_offset)
            ivv_result = ivv_group['data'][:]
            ivv_expected_result = np.arange(64 * self.ivv_frequency,
                                            dtype=np.float)
            self.assertEqual(list(ivv_result), list(ivv_expected_result))
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(64 * self.wow_frequency,
                                            dtype=np.float)
            self.assertEqual(list(wow_result), list(wow_expected_result))
            # 'DME' - 0.25Hz parameter.
            dme_group = hdf_file['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.arange(64 * self.dme_frequency,
                                            dtype=np.float)
            self.assertEqual(list(dme_result), list(dme_expected_result))
            self.assertEqual(hdf_file.attrs['duration'], 64)

    @unittest.skip('Rewrite this test to use the API to prepare the test data and to verify it')
    def test_write_segment__start_only(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(50, None)
        frame_start = 48  # 48 is nearest frame boundary rounded down
        dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=4)
        self.assertEqual(dest, self.out_path)

        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            ivv_group = hdf_file['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['offset'],
                             self.ivv_supf_offset)
            # mask = merge_masks(ivv_group['submasks'][:])
            mask = False
            ivv_result = np.ma.masked_array(ivv_group['data'][:], mask=mask)
            ivv_expected_result = np.arange(
                frame_start * self.ivv_frequency,
                self.data_secs * self.ivv_frequency,
                dtype=np.float)
            ivv_expected_result = np.ma.masked_array(
                ivv_expected_result, mask=[True] * 2 + [False] * 78)
            self.assertEqual(ivv_result.tolist(), ivv_expected_result.tolist())
            # 'WOW' - 4Hz parameter.
            wow_group = hdf_file['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(
                frame_start * self.wow_frequency,
                self.data_secs * self.wow_frequency,
                dtype=np.float)
            self.assertEqual(wow_result.tolist(), wow_expected_result.tolist())
            # 'DME' - 0.25Hz parameter.
            dme_group = hdf_file['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.arange(12, 32, dtype=np.float)
            self.assertEqual(dme_result.tolist(), dme_expected_result.tolist())
            self.assertEqual(hdf_file.attrs['duration'], 80)

    def test_write_segment__from_path__no_dest(self):
        '''
        Tests that the destination filename is constructed correctly when source filename is pased to write_segment.
        '''
        segment = slice(None, 70)
        with self.assertWarns(DeprecationWarning):
            dest = write_segment(self.hdf_path, segment).path
        expected_dest_basename = os.path.splitext(self.hdf_path)[0] + '.000.hdf5'
        expected = os.path.join(os.path.dirname(self.hdf_path), expected_dest_basename)
        self.assertEqual(dest, expected)

    def test_write_segment__from_object__no_dest(self):
        '''
        Tests that the destination filename is constructed correctly.
        '''
        segment = slice(None, 70)
        with fda.open(self.hdf_path) as fdf:
            with self.assertWarns(DeprecationWarning):
                dest = write_segment(fdf, segment).path

        expected_dest_basename = os.path.splitext(self.hdf_path)[0] + '.000.hdf5'
        expected = os.path.join(os.path.dirname(self.hdf_path), expected_dest_basename)
        self.assertEqual(dest, expected)

    def test_write_segment__submasks(self):
        '''
        Ensure a warning is raised when `submasks` argument is passed.
        '''
        segment = slice(None, 70)
        with fda.open(self.hdf_path) as fdf:
            with self.assertWarns(DeprecationWarning):
                write_segment(fdf, segment, submasks=['sub1', 'sub2'])

    def test_write_segment__stop_only(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        segment = slice(None, 70)
        with self.assertWarns(DeprecationWarning):
            dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=4).path
        self.assertEqual(dest, self.out_path)

        frame_boundary_segment = slice(None, 72)

        with h5py.File(dest, 'r') as hdf_file:
            # 'IVV' - 1Hz parameter.
            series = hdf_file
            ivv_group = series['IVV']
            self.assertEqual(ivv_group.attrs['frequency'],
                             self.ivv_frequency)
            self.assertEqual(ivv_group.attrs['offset'], self.ivv_supf_offset)
            ivv_result = ivv_group['data'][:]
            ivv_expected_result = np.arange(
                0, frame_boundary_segment.stop * self.ivv_frequency,
                dtype=np.float)
            self.assertItemsEqual(ivv_result, ivv_expected_result)
            # 'WOW' - 4Hz parameter.
            wow_group = series['WOW']
            self.assertEqual(wow_group.attrs['frequency'],
                             self.wow_frequency)
            wow_result = wow_group['data'][:]
            wow_expected_result = np.arange(
                0, frame_boundary_segment.stop * self.wow_frequency,
                dtype=np.float)
            self.assertTrue(list(wow_result), list(wow_expected_result))
            # 'DME' - 0.25Hz parameter.
            dme_group = series['DME']
            self.assertEqual(dme_group.attrs['frequency'],
                             self.dme_frequency)
            dme_result = dme_group['data'][:]
            dme_expected_result = np.arange(0, 18, dtype=np.float)
            self.assertEqual(list(dme_result), list(dme_expected_result))
            self.assertEqual(hdf_file.attrs['duration'], 72)

    def test_write_segment__all_data(self):
        '''
        Tests that the correct segment of the dataset within the path matching
        'series/<Param Name>/data' defined by the slice has been written to the
        destination file while other datasets and attributes are unaffected.
        Slice has a start and stop.
        '''
        def test_hdf(dest):
            with h5py.File(dest, 'r') as hdf_file:
                # 'IVV' - 1Hz parameter.
                series = hdf_file
                ivv_group = series['IVV']
                self.assertEqual(ivv_group.attrs['frequency'],
                                 self.ivv_frequency)
                self.assertEqual(ivv_group.attrs['offset'],
                                 self.ivv_supf_offset)
                ivv_result = ivv_group['data'][:]
                self.assertTrue(all(ivv_result == self.ivv_data))
                # 'WOW' - 4Hz parameter.
                wow_group = series['WOW']
                self.assertEqual(wow_group.attrs['frequency'],
                                 self.wow_frequency)
                wow_result = wow_group['data'][:]
                self.assertTrue(all(wow_result == self.wow_data))
                # 'DME' - 0.25Hz parameter.
                dme_group = series['DME']
                self.assertEqual(dme_group.attrs['frequency'],
                                 self.dme_frequency)
                dme_result = dme_group['data'][:]
                self.assertTrue(all(dme_result == self.dme_data))
                self.assertEqual(hdf_file.attrs['duration'], self.data_secs)

        segment = slice(None)
        with self.assertWarns(DeprecationWarning):
            dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=4).path
        self.assertEqual(dest, self.out_path)
        test_hdf(dest)
        with self.assertWarns(DeprecationWarning):
            dest = write_segment(self.hdf_path, segment, dest=self.out_path, boundary=64).path
        self.assertEqual(dest, self.out_path)
        test_hdf(dest)

    def tearDown(self):
        try:
            os.remove(self.hdf_path)
        except OSError as err:
            if err.errno != errno.ENOENT:
                raise
        try:
            os.remove(self.out_path)
        except OSError as err:
            if err.errno != errno.ENOENT:
                raise


if __name__ == '__main__':
    unittest.main()