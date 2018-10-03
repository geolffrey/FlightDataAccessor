# Tasks to perform:
# [*] Move all parameters from 'series' group to top-level.
# [*] Remove 'mask' dataset and rename 'submasks' dataset to 'mask'.
# [*] Add 'mask' as a new 'legacy' submask if not equal to flattened submasks.
# [*] Remove 'levels' group and all groups and datasets beneath.
# [*] Remove obsolete global attributes, e.g. 'hdfaccess_version'
# [*] Fix up the data type for the 'data' dataset where not <f8.
# [*] Re-serialise all attributes stored as json to remove whitespace.
# [ ] Re-serialise all attributes stored as json/gzip/base64 to remove whitespace.
# [*] Store global attributes such as 'superframe_present' as a boolean.
# [*] Discard unused global attributes? e.g. 'description'?
# [*] Update global 'version' attribute.

import h5py
import numpy as np


# TODO: Add deprecation warnings!
class Compatibility(object):

    # Legacy functions for handling pickled global attributes:
    get_attr = lambda self, name, default=None: getattr(self, name, default)
    set_attr = lambda self, name, value: setattr(self, name, value)

    # Legacy functions for retrieving/storing/deleting parameter data:
    get_param = lambda self, *args, **kwargs: self.get_parameter(*args, **kwargs)
    set_param = lambda self, *args, **kwargs: self.set_parameter(*args, **kwargs)
    get_params = lambda self, *args, **kwargs: self.get_parameters(*args, **kwargs)
    delete_params = lambda self, *args, **kwargs: self.delete_parameters(*args, **kwargs)

    # Legacy functions for looking up different groups of parameter names:
    get_param_list = lambda self: self.keys()
    valid_param_names = lambda self: self.keys(valid_only=True)
    valid_lfl_param_names = lambda self: self.keys(valid_only=True, subset='lfl')
    lfl_keys = lambda self: self.keys(subset='lfl')
    derived_keys = lambda self: self.keys(subset='derived')

    # Legacy properties for handling custom tweaks to global attributes:
    # - superframe_present
    # - version

    @property
    def hdf(self):
        return self.file  # FIXME: What about accessing ['series']?

    def upgrade(self, filename):
        if self.file.attrs.get('version') >= self.VERSION:
            raise ValueError('The FlightDataFile is in the latest format!')

        with self.__class__(filename) as new_fdf:
            new_fdf.set_parameters(self.values())
            for name, value in self.file.attrs.items():
                if name == 'version':
                    continue

                new_fdf.file.attrs.create(name, value)
