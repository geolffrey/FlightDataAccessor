import numpy as np


SAMPLES_PER_BUCKET = 2


def masked_invalid(data):
    try:
        return np.ma.masked_invalid(data)
    except TypeError:
        return data # isfinite not supported for input type, e.g. string


def downsample(data, bucket_size, point_size=1):
    '''
    Data-processing helper for downsampling consecutive data.  bucket_size is number of consecutive points to coalesce
    into one bucket. point_size is the number of values for each point (used when downsampling already downsampled
    data). Result is array (sorted on x) with beginning1, min1, max1, beginning2, ...
    '''

    if bucket_size == 1 and point_size > 1:  # easy
        return data

    if point_size > 1:
        # actually, for already downsampled data we could just compute minimum (similar for max) over the already
        # computed minimums, however this is a bit more complicated and numpy actually seems to be a bit faster at
        # computing minimums over the whole data set instead of over a sparse slice ala [::3]; so we just handle
        # already downsampled data as if the buckets were larger
        bucket_size *= point_size

    # unfortunately, numpy can't deal with irregular array sizes, so we need to split the data set
    remainder = len(data) % bucket_size
    regular_part = masked_invalid(data[:len(data) - remainder])

    # first calculate the indexes of all the numbers we want
    minimums = regular_part.reshape(-1, bucket_size).argmin(axis=1)
    maximums = regular_part.reshape(-1, bucket_size).argmax(axis=1)
    if remainder:
        remainder_part = masked_invalid(data[len(data) - remainder:])
        minimums = np.concatenate((minimums, [remainder_part.argmin()]))
        maximums = np.concatenate((maximums, [remainder_part.argmax()]))

    beginnings = np.arange(0, len(minimums) * bucket_size, bucket_size, dtype=minimums.dtype)

    # zip the indexes together
    indexes = np.column_stack((minimums + beginnings, maximums + beginnings)).reshape((-1,))
    indexes.sort()
    return data[indexes]
