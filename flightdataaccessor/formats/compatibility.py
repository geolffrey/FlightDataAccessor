"""Compatibility features."""

import inspect


def open(source, **kwargs):
    """Open the source object."""
    from . import base
    from . import hdf

    if isinstance(source, str):
        # path to a file
        return hdf.FlightDataFile(source, **kwargs)

    elif isinstance(source, base.FlightDataFormat):
        # already instanciated Flight Data Format object
        return source

    elif inspect.isclass(source) and issubclass(source, base.FlightDataFormat):
        # class used to create an empty Flight Data Format object
        return source(**kwargs)

    else:
        raise ValueError('Type of passed argument `source` not supported: %s %s' % (type(source), source))
