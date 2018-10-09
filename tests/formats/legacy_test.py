import os
import shutil
import tempfile

import unittest

from flightdataaccessor.formats.hdf import FlightDataFile
from flightdataaccessor.formats.legacy import REMOVE_GLOBAL_ATTRIBUTES, RENAME_GLOBAL_ATTRIBUTES


class CompatibilityTest(unittest.TestCase):
    test_fn = 'data/flight_data_v2.hdf5'

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        curr_dir = os.path.dirname(__file__)
        self.old_fp = os.path.join(self.tempdir, os.path.basename(self.test_fn))
        self.new_fp = self.old_fp.replace('.hdf5', '-new.hdf5')
        shutil.copy(os.path.join(curr_dir, self.test_fn), self.old_fp)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def upgrade_test(self):
        """
        Take file in old format and upgrade it.

        Check if everything is in right place afterwards.
        """
        with FlightDataFile(self.old_fp) as fdf:
            # load file in "old" format
            old_params = fdf.keys()
            fdf.upgrade(self.new_fp)

        with FlightDataFile(self.new_fp) as fdf:
            # load the upgraded file
            global_attr_names = fdf.file.attrs.keys()
            for attr_name in REMOVE_GLOBAL_ATTRIBUTES:
                self.assertNotIn(attr_name, global_attr_names)

            for old_attr_name, new_attr_name in RENAME_GLOBAL_ATTRIBUTES.items():
                self.assertNotIn(old_attr_name, global_attr_names)
                self.assertIn(new_attr_name, global_attr_names)

            new_params = fdf.keys()

        self.assertEquals(old_params, new_params)
