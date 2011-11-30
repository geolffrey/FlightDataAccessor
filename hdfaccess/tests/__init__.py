import unittest


def master_suite():
    from skeleton.tests import testSkeleton 
    suite = unittest.TestSuite()
    suite.addTest(testSkeleton.suite())
    return suite


def suite():
    suite = unittest.TestSuite()
    suite.addTest(master_suite())
    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
