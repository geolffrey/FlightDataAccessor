"""
Compatibility features.
"""
import contextlib


@contextlib.contextmanager
def open(source, **kwargs):
    """Open the source object."""
    from . import base
    from . import hdf

    if isinstance(source, str):
        # path to a file
        with hdf.FlightDataFile(source, **kwargs) as fdf:
            yield fdf

    elif isinstance(source, base.FlightDataFormat):
        # already instanciated Flight Data Format object
        yield source

    elif issubclass(source, base.FlightDataFormat):
        # class used to create an empty Flight Data Format object
        with source(**kwargs) as fdf:
            yield fdf
