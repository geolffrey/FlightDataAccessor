"""
Legacy factory to open FlightDataFile.

The contents of this module is deprecated.
"""
from __future__ import print_function

from deprecation import deprecated

from .formats import compatibility, hdf

# XXX: refactor dependent code and remove
HDFACCESS_VERSION = hdf.CURRENT_VERSION


@deprecated(details='Use flightdataaccessor.open() or format classes instead')
def hdf_file(*args, **kwargs):
    """Open and return FlightDataFile."""
    if 'mode' not in kwargs:
        create = kwargs.pop('create', None)
        read_only = kwargs.pop('read_only', None)
        if read_only and create:
            raise ValueError('Creation of a new file in read only mode is not supported')
        if read_only:
            mode = 'r'
        elif create:
            mode = 'x'
        else:
            # legacy default mode
            mode = 'r+'
        kwargs['mode'] = mode
    return compatibility.open(*args, **kwargs)


def print_hdf_info(hdf_file):
    """Open FlightDataFile information."""
    for group_name, group in hdf_file.hdf['series'].items():
        print('[%s]' % group_name)
        print('Frequency:', group.attrs['frequency'])
        print(group.attrs.items())
        print('Offset:', group.attrs['supf_offset'])
        print('Number of recorded values:', len(group['data']))


if __name__ == '__main__':
    import sys
    print_hdf_info(hdf_file(sys.argv[1]))
    sys.exit()
    file_path = 'AnalysisEngine/resources/data/hdf5/flight_1626325.hdf5'
    with hdf_file(file_path) as hdf:
        print_hdf_info(hdf)
