import os
import shutil
import tempfile
import unittest

from flightdataaccessor.formats.hdf import FlightDataFile
from flightdataaccessor.formats.legacy import REMOVED_GLOBAL_ATTRIBUTES, RENAMED_GLOBAL_ATTRIBUTES


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

    def test_upgrade(self):
        """
        Take file in old format and upgrade it.

        Check if everything is in right place afterwards.
        """
        with FlightDataFile(self.old_fp) as old_fdf:
            old_fdf.upgrade(self.new_fp)

            with FlightDataFile(self.new_fp) as new_fdf:
                self.assertEqual(new_fdf.version, FlightDataFile.VERSION)
                old_global_attr_names = old_fdf.file.attrs.keys()
                new_global_attr_names = new_fdf.file.attrs.keys()
                for old_attr_name in old_global_attr_names:
                    if old_attr_name in REMOVED_GLOBAL_ATTRIBUTES:
                        self.assertNotIn(old_attr_name, new_global_attr_names)
                    elif old_attr_name in RENAMED_GLOBAL_ATTRIBUTES:
                        new_attr_name = RENAMED_GLOBAL_ATTRIBUTES[old_attr_name]
                        self.assertNotIn(old_attr_name, new_global_attr_names)
                        self.assertIn(new_attr_name, new_global_attr_names)

                for parameter in old_fdf.values():
                    self.assertIn(parameter.name, new_fdf)
                    if parameter.submasks:
                        self.assertTrue(new_fdf[parameter.name].submasks)
