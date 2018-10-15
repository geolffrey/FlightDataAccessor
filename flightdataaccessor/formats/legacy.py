from deprecation import deprecated


REMOVE_GLOBAL_ATTRIBUTES = [
    'achieved_flight_record', 'aircraft_info', 'tailmark', 'starttime', 'endtime',
]

RENAME_GLOBAL_ATTRIBUTES = {
    'analysis_version': 'version_analyzer',
    'hdfaccess_version': 'version',
    'start_timestamp': 'timestamp',
}


class Compatibility(object):
    """Support for legacy properties and methods"""

    # Legacy functions for handling pickled global attributes:
    @deprecated(details='Use standard attribute read instead')
    def get_attr(self, name, default=None):
        return getattr(self, name, default=default)

    @deprecated(details='Use standard attribute assignment instead')
    def set_attr(self, name, value):
        return setattr(self, name, value)

    # Legacy functions for retrieving/storing/deleting parameter data:
    @deprecated(details="Use `get_parameter()` instead")
    def get_param(self, *args, **kwargs):
        return self.get_parameter(*args, **kwargs)

    @deprecated(details='Use `get_parameter_limits()` instead')
    def get_param_limits(self, *args, **kwargs):
        return self.get_parameter_limits(*args, **kwargs)

    @deprecated(details="Use `set_parameter()` instead")
    def set_param(self, *args, **kwargs):
        return self.set_parameter(*args, **kwargs)

    @deprecated(details='Use `set_parameter_limits()` instead')
    def set_param_limits(self, *args, **kwargs):
        return self.set_parameter_limits(*args, **kwargs)

    @deprecated(details='Use `set_parameter_invalid()` instead')
    def set_invalid(self, *args, **kwargs):
        return self.set_parameter_invalid(*args, **kwargs)

    @deprecated(details='Use `get_parameters()` instead')
    def get_params(self, param_names=None, **kwargs):
        return self.get_parameters(names=param_names, **kwargs)

    @deprecated(details="Use `delete_parameters()` instead")
    def delete_params(self, *args, **kwargs):
        return self.delete_parameters(*args, **kwargs)

    # Legacy functions for looking up different groups of parameter names:
    @deprecated(details="Use `keys()` instead")
    def get_param_list(self):
        return self.keys()

    @deprecated(details="Use `keys(valid_only=True)` instead")
    def valid_param_names(self):
        return self.keys(valid_only=True)

    @deprecated(details="Use `keys(valid_only=True, subset='source')` instead")
    def valid_lfl_param_names(self):
        return self.keys(valid_only=True, subset='source')

    @deprecated(details="Use `keys(subset='source')` instead")
    def lfl_keys(self):
        return self.keys(subset='source')

    @deprecated(details="Use `keys(subset='derived')` instead")
    def derived_keys(self):
        return self.keys(subset='derived')

    # Legacy properties for handling custom tweaks to global attributes:
    # - superframe_present
    # - version

    @property
    @deprecated(details='Use `file` instead')
    def hdf(self):
        return self.file  # FIXME: What about accessing ['series']?

    def upgrade(self, filename):
        if self.file.attrs.get('version') >= self.VERSION:
            raise ValueError('The FlightDataFile is in the latest format!')

        with self.__class__(filename, mode='x') as new_fdf:
            new_fdf.set_parameters(self.values())
            for name, value in self.file.attrs.items():
                if name in REMOVE_GLOBAL_ATTRIBUTES or name == 'version':
                    continue

                name = RENAME_GLOBAL_ATTRIBUTES.get(name, name)
                new_fdf.file.attrs.create(name, value)
