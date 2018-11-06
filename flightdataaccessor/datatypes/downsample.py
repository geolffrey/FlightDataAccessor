import numpy as np


SAMPLES_PER_BUCKET = 2


def masked_invalid(data):
    try:
        return np.ma.masked_invalid(data)
    except TypeError:
        return data  # isfinite not supported for input type, e.g. string


# XXX: copy-paste from analysis_engine.library. Needs to be sorted out with FlightDataAnalyzer refactoring to create a
# utilities package with common functionality used by many repositories to avoid cross-dependencies.
def most_common_value(array, threshold=None):
    '''
    Find the most repeating non-negative valid value within an array. Works
    with mapped arrays too as well as arrays of strings.

    :param array: Array to count occurrences of values within
    :type array: np.ma.array
    :param threshold: return None if number of most common value occurrences is
        less than this
    :type threshold: float
    :returns: most common value in array
    '''
    if isinstance(array, np.ma.MaskedArray):
        array = array.compressed()
    values, counts = np.unique(array, return_counts=True)
    if not counts.size:
        return None

    i = np.argmax(counts)
    value, count = values[i], counts[i]

    if threshold is not None and count < array.size * threshold:
        return None

    if hasattr(array, 'values_mapping'):
        return array.values_mapping[value]
    else:
        return value


def downsample_most_common_value(data, width):
    """Downsample non numeric data.

    In case of non numeric values (as well as MappedArrays) we want to return most common values instead of min/max
    pairs for each bucket.

    The returned array has one sample per bucket.
    """
    # XXX: move the functionality to independent package: FlightDataUtilities?
    from .parameter import MappedArray

    array = np.ma.asarray(data)

    bucket_size = array.size // width
    remainder = len(array) % bucket_size
    regular_part = masked_invalid(array[:len(array) - remainder])
    samples = []
    for bucket in regular_part.reshape(-1, bucket_size):
        value = most_common_value(bucket)
        if value is None:
            value = np.ma.masked
        samples.append(value)

    if remainder:
        remainder_part = masked_invalid(array[len(array) - remainder:])
        value = most_common_value(remainder_part)
        if value is None:
            value = np.ma.masked
        samples.append(value)

    if isinstance(data, MappedArray):
        return MappedArray(samples, values_mapping=data.values_mapping)
    return np.ma.array(samples, dtype=np.asarray(data).dtype)


def downsample(data, width):
    """Downsample data.

    bucket_size is number of consecutive points to coalesce into one bucket. point_size is the number of values for
    each point (used when downsampling already downsampled data). Result is array (sorted on x) with min1, max1, min2,
    max2, ...

    The returned array has 2 samples per bucket.
    """
    from .parameter import MappedArray

    array = np.ma.asarray(data)
    if isinstance(data, MappedArray) or array.dtype.kind not in 'uif':
        return downsample_most_common_value(data, width)

    # 2 samples per bucket
    bucket_size = 2 * array.size // width
    # unfortunately, numpy can't deal with irregular array sizes, so we need to split the data set
    remainder = len(array) % bucket_size
    regular_part = masked_invalid(array[:len(array) - remainder]).reshape(-1, bucket_size)

    # first calculate the indexes of all the numbers we want
    minimums = regular_part.argmin(axis=1)
    maximums = regular_part.argmax(axis=1)
    if remainder:
        remainder_part = masked_invalid(array[len(array) - remainder:])
        minimums = np.concatenate((minimums, [remainder_part.argmin()]))
        maximums = np.concatenate((maximums, [remainder_part.argmax()]))

    beginnings = np.arange(0, len(minimums) * bucket_size, bucket_size, dtype=minimums.dtype)

    # zip the indexes together
    indexes = np.column_stack((minimums + beginnings, maximums + beginnings)).reshape((-1,))
    indexes.sort()
    return array[indexes]
