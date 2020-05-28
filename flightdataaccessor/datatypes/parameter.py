#------------------------------------------------------------------------------
# Parameter container Class
# =========================
'''
Parameter container class.
'''
import copy
import math

import numpy as np

from flightdatautilities.array_operations import merge_masks

from .array import MappedArray, MaskError, ParameterArray, ParameterSubmasks
from .downsample import SAMPLES_PER_BUCKET, downsample
from .legacy import Legacy

# XXX: deprecate!
NO_MAPPING = MappedArray.NO_MAPPING


class Parameter(Legacy):
    array = ParameterArray()

    def __init__(self, name, array=[], values_mapping=None, frequency=1, offset=0, arinc_429=None, invalid=None,
                 invalidity_reason=None, unit=None, data_type=None, source=None, source_name=None, description='',
                 submasks=None, limits=None, compress=False, **kwargs):
        '''
        :param name: Parameter name
        :type name: String
        :param array: Masked array of data for the parameter.
        :type array: np.ma.masked_array
        :param values_mapping: Values mapping of a multi-state parameter.
        :type values_mapping: dict or None
        :param frequency: Sample Rate / Frequency / Hz
        :type frequency: float
        :param offset: The offset of the parameter in seconds within a
            superframe.
        :type offset: float
        :param arinc_429: Whether or not the parameter stores ARINC 429 data.
        :type arinc_429: bool or None
        :param invalid: Whether or not the parameter has been marked as
            invalid.
        :type invalid: bool or None
        :param invalidity_reason: The reason why the parameter was marked as
            invalid.
        :type invalidity_reason: str or None
        :param units: The unit of measurement the parameter is recorded in.
        :type units: str or None
        :param lfl: Whether or not the parameter is from the LFL or derived.
        :type lfl: bool or None
        :param source_name: The original name of the parameter.
        :type source_name: str or None
        :param description: Description of the parameter.
        :type description: str
        :param submasks: Default value is None to avoid kwarg default being mutable.
        '''
        self.name = name

        if values_mapping or not getattr(self, 'values_mapping', None):
            self.values_mapping = values_mapping or {}

        # ensure frequency is stored as a float
        self.frequency = float(frequency)
        self.offset = offset
        self.arinc_429 = arinc_429
        self.unit = unit
        self.data_type = data_type
        self.source = source if source else 'lfl'
        self.lfl = self.source == 'lfl'  # Fixme: remove when source is fully implemented
        self.source_name = source_name
        self.description = description
        self.invalid = invalid
        self.invalidity_reason = invalidity_reason
        self.limits = limits
        self.compress = compress

        submasks = {k: np.array(v, dtype=np.bool) for k, v in submasks.items()} if submasks else {}
        self.submasks = ParameterSubmasks(submasks, compress=self.compress)
        self.validate_mask(array, submasks)
        self.array = array

    def __repr__(self):
        return "%s %sHz %.2fsecs" % (self.name, self.frequency, self.offset)

    def is_compatible(self, parameter=None, name=None, frequency=None, offset=None, unit=None):
        """Check if another parameter is compatible with this one."""
        if parameter:
            name = parameter.name
            frequency = parameter.frequency
            offset = parameter.offset
            unit = parameter.unit

        return (
            self.name == name
            and self.frequency == frequency
            and self.offset == offset
            and self.unit == unit
        )

    @property
    def default_submask_name(self):
        """ Name of a default submask.

        A default submask is created when the parameter is populated with MaskedArray which contains masked values that
        are incompatible with corresponding submasks or if MaskedArray is passed to parameter without any submasks."""
        # XXX: Node should have a default source value
        source = getattr(self, 'source', 'lfl')
        sources = {
            'lfl': 'padding',
            'derived': 'derived',
        }
        return sources.get(source, 'auto')

    @property
    def duration(self):
        """Calculate the duration of data."""
        return len(self.array) / self.frequency

    @property
    def raw_array(self):
        if getattr(self, 'values_mapping', None):
            return self.array.raw
        return self.array

    def set_array(self, array, submasks):
        self.validate_mask(array, submasks)
        self.array = array
        self.submasks = ParameterSubmasks(submasks, compress=self.compress)

    def get_array(self, submask=None):
        '''
        Get the Parameter's array with an optional submask substituted for the
        mask.

        :param submask: Name of submask to return with the array.
        :type submask: str or None
        '''
        if not submask:
            return self.array
        if submask not in self.submasks:
            return
        if isinstance(self.array, MappedArray):
            return MappedArray(self.array.data, mask=self.submasks[submask].copy(),
                               values_mapping=self.array.values_mapping)
        else:
            return np.ma.MaskedArray(self.array.data, mask=self.submasks[submask].copy())

    def downsample(self, width, start_offset=None, stop_offset=None, mask=True):
        """Downsample data in range to fit in a window of given width."""
        start_ix = int(start_offset * self.frequency) if start_offset else 0
        stop_ix = int(stop_offset * self.frequency) if stop_offset else self.array.size
        sliced = self.array[start_ix:stop_ix]
        if not mask:
            sliced = sliced.data
        if sliced.size <= width:
            return sliced, None

        bucket_size = sliced.size // width
        if bucket_size > 1:
            downsampled = downsample(sliced, width)
            return downsampled, bucket_size
        else:
            return sliced, bucket_size

    def zoom(self, width, start_offset=0, stop_offset=None, mask=True, timestamps=False):
        """Zoom out to display the data in range in a window of given width.

        Optionally combine the data with timestamp information (in miliseconds).

        This method is designed for use in data visualisation."""
        downsampled, bucket_size = self.downsample(width, start_offset=start_offset, stop_offset=stop_offset, mask=mask)
        if not timestamps:
            return downsampled

        interval = 1000 * (1 if bucket_size is None else bucket_size / SAMPLES_PER_BUCKET) / self.frequency
        timestamps = 1000 * (self.offset + start_offset) + interval * np.arange(downsampled.size)
        return np.ma.dstack((timestamps, downsampled))[0]

    def slice(self, sl):
        """Return a copy of the parameter with all the data sliced to given slice."""
        clone = copy.deepcopy(self)
        clone.set_array(self.array[sl], submasks={k: v[sl] for k, v in self.submasks.items()})
        return clone

    def trim(self, start_offset=0, stop_offset=None, pad_subframes=4):
        """Return a copy of the parameter with all the data trimmed to given window in seconds.

        Optionally align the window to pad_subframes blocks of subframes (defaults to a single frame)which is useful
        for splitting segments."""
        if start_offset is None:
            start_offset = 0
        if stop_offset is None:
            stop_offset = self.array.size / self.frequency

        unmasked_start_offset = start_offset
        unmasked_stop_offset = stop_offset
        if pad_subframes:
            start_offset = pad_subframes * math.floor(start_offset / pad_subframes) if start_offset else 0
            stop_offset = pad_subframes * math.ceil(stop_offset / pad_subframes)
        start_ix = math.floor(start_offset * self.frequency) if start_offset else 0
        stop_ix = math.ceil(stop_offset * self.frequency) if stop_offset else self.array.size
        clone = self.slice(slice(start_ix, stop_ix))
        if pad_subframes and stop_ix > self.array.size:
            # more data was requested than available and padding was requested
            # ensure that the clone has a padding submask
            clone.update_submask('padding', np.zeros(clone.array.size, dtype=np.bool))
            padding = np.ma.zeros(stop_ix - self.array.size, dtype=np.bool)
            padding.mask = True
            padding_submasks = {k: np.zeros(padding.size, dtype=np.bool) for k in self.submasks}
            padding_submasks['padding'] = np.ones(padding.size, dtype=np.bool)
            clone.extend(padding, submasks=padding_submasks)

        # mask the areas outside of requested slice
        if unmasked_start_offset > start_offset or stop_offset > unmasked_stop_offset:
            requested_duration = unmasked_stop_offset - unmasked_start_offset
            padding = np.zeros(len(clone.array), dtype=np.bool)
            padding_at_start = int(unmasked_start_offset * self.frequency - start_ix)
            padding_at_end = int(padding_at_start + requested_duration * self.frequency)
            padding[:padding_at_start] = True
            padding[padding_at_end:] = True
            clone.update_submask('padding', padding)

        return clone

    def extend(self, data, submasks=None):
        """Extend the parameter's data."""
        if isinstance(data, Parameter):
            if not self.is_compatible(data):
                raise ValueError('Parameter passed to extend() is not compatible')
            if submasks:
                raise MaskError('`submasks` argument is not accepted if a Parameter is passed to extend()')

        array, submasks = self.build_array_submasks(data, submasks=submasks)

        for name in submasks:
            if name not in self.submasks:
                # add an empty submask up to this point
                self.submasks[name] = np.zeros(len(self.array), dtype=np.bool8)
            self.submasks[name] = np.ma.concatenate([self.submasks[name], submasks[name]])

        array = np.ma.asanyarray(array)
        if isinstance(self.array, MappedArray):
            if array.dtype.type is np.str_:
                state = {v: k for k, v in self.values_mapping.items()}
                array = [state.get(x, None) for x in array]
            array = MappedArray(np.ma.asanyarray(array), values_mapping=self.values_mapping)

        self.array = np.ma.append(self.array, array)

    # Submasks handling
    def validate_mask(self, array=None, submasks=None, strict=False):
        """Verify if combined submasks are equivalent to array.mask.

        In default mode if submasks are not defined the array's mask is consoidered valid.

        In strict mode the mask and submasks mustr always be equivalent.
        """
        if array is None and submasks is None:
            array = self.array
            submasks = {k: v for k, v in self.submasks.items() if len(v)}

        if not submasks:
            if strict and np.any(np.ma.getmaskarray(array)):
                raise MaskError("Submasks are not defined and array is masked")
            return True

        # we want to handle "default" submask no matter if already exists or is added
        old_submask_names = set(self.submasks.keys()) | {self.default_submask_name}
        new_submask_names = set(submasks.keys()) | {self.default_submask_name}
        if new_submask_names != old_submask_names:
            raise MaskError("Submask names don't match the stored submasks")
        for submask_name, submask in submasks.items():
            if len(submask) != len(array):
                raise MaskError("Submasks don't have the same length as the array")
        if isinstance(array, np.ma.MaskedArray):
            mask = self.combine_submasks(submasks, array)
            if not np.all(mask == np.ma.getmaskarray(array)):
                raise MaskError('Submasks are not equivalent to array.mask')

        return True

    def combine_submasks(self, submasks=None, array=None):
        '''
        Combine submasks into a single OR'd mask.

        If optional array is passed it's mask will be returned if submasks are empty.

        :returns: Combined submask.
        :rtype: np.array
        '''
        if submasks is None:
            submasks = self.submasks
            array = self.array

        if submasks:
            return merge_masks(list(submasks.values()))
        else:
            return np.zeros(len(array), dtype=np.bool)

    def submasks_from_array(self, array, submasks=None):
        """Build submasks compatible with parameter for the passed array of data.

        The idea is to allow expansion of data with an array of values and keep submasks contents consistent.

        The code assumes that array and submasks are validated and we only need to fill the gaps.
        """
        submasks = submasks or {}
        for name in self.submasks:
            if name not in submasks:
                submasks[name] = np.zeros(len(array), dtype=np.bool)

        mask_array = np.ma.getmaskarray(array)
        if np.any(mask_array != self.combine_submasks(submasks, array)):
            submasks[self.default_submask_name] = mask_array

        if self.data_type == 'Multi-state':
            submasks['invalid_states'] = ~np.isin(array, list(self.values_mapping))

        return submasks

    def build_array_submasks(self, data, submasks=None):
        """Build array and submasks from passed data.

        The data is normalised to provide formats compatible with the parameter."""
        if isinstance(data, Parameter):
            array = data.array
            submasks = data.submasks
        else:
            array = data

        self.validate_mask(array=array, submasks=submasks)
        submasks = self.submasks_from_array(array, submasks)

        return array, submasks

    def update_submask(self, name, mask, merge=True):
        """Update a submask.

        If merge is True the submask is updated (logical OR) with the passed value, otherwise it's replaced.
        Parameter.array.mask is updated automatically to stay in sync.

        If submask with given name does not exist, it will be created."""
        if merge or name not in self.submasks:
            self.submasks[name] = mask
        else:
            self.submasks[name] |= mask

        array = self.array.copy()
        array.mask = self.combine_submasks()
        self.set_array(array, self.submasks)
