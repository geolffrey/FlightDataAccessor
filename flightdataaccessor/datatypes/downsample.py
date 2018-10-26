import numpy as np

from analysis_engine import library


SAMPLES_PER_BUCKET = 2


def masked_invalid(data):
    try:
        return np.ma.masked_invalid(data)
    except TypeError:
        return data  # isfinite not supported for input type, e.g. string


def downsample_most_common_value(data, width):
    """Downsample non numeric data.

    In case of non numeric values (as well as MappedArrays) we want to return most common values instead of min/max
    pairs for each bucket.

    The returned array has one sample per bucket.
    """
    from .parameter import MappedArray

    array = np.ma.asarray(data)

    bucket_size = array.size // width
    remainder = len(array) % bucket_size
    regular_part = masked_invalid(array[:len(array) - remainder])
    samples = []
    for bucket in regular_part.reshape(-1, bucket_size):
        value = library.most_common_value(bucket)
        if value is None:
            value = np.ma.masked
        samples.append(value)

    if remainder:
        remainder_part = masked_invalid(array[len(array) - remainder:])
        value = library.most_common_value(remainder_part)
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
