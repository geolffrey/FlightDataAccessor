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
        
        a.mask = False
        a[:3] = np.ma.masked
        self.assertEqual(list(a.raw.mask), [True, True, True, False])
        a[:] = np.ma.array([3,3,3,3], mask=[False, False, True, True])
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
        mapping = {1: 'one', 2: 'two', 3: 'three'}
        mask = [False, True, False, True]
        arr = np.ma.MaskedArray(values, mask)
        a = MappedArray(arr, values_mapping=mapping)        
        # ensure string vals is within repr
        print a.__repr__()
        self.assertTrue('one' in a.__repr__())
        
    def test_getitem_filters_boolean_array(self):
        ma = MappedArray(np.ma.arange(4,-1,step=-1), values_mapping={1:'one', 2:'two'})
        
        # boolean returned where: array == value
        #                                 4       3      2     >1<     0
        self.assertEqual(list(ma == 1), [False, False, False, True, False])
        
        # Nice to Have : Overide == for state
        # boolean returned where: array == 'state'
        #                                     4       3     >2<      1     0
        self.assertEqual(list(ma == 'two'), [False, False, True,  False, False])
        
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
