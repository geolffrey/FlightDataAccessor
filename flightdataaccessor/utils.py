from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import shutil
import six

from deprecation import deprecated

from flightdatautilities.array_operations import merge_masks
from flightdatautilities.filesystem_tools import copy_file

from flightdataaccessor.formats.hdf import FlightDataFile


def _copy_attrs(source_group, target_group, deidentify=False):
    '''
    While the library can recursively copy groups and datasets, there does
    not seem to be a simple way to copy all of a group's attributes at once.
    '''
    for key, value in source_group.attrs.items():
        # TODO: We are planning to upgrade the file format to remove cruft.
        #       Remove obsolete attribute names when we have upgraded all files.
        if deidentify and key in ('aircraft_info', 'tailmark'):
            continue
        target_group.attrs[key] = value


@deprecated(details='Use FlightDataFile.concatenate() instead')
def concat_hdf(hdf_paths, dest=None):
    '''
    Takes in a list of HDF file paths and concatenates the parameter
    datasets which match the path 'series/<Param Name>/data'. The first file
    in the list of paths is the template for the output file, with only the
    'series/<Param Name>/data' datasets being replaced with the concatenated
    versions.

    :param hdf_paths: File paths.
    :type hdf_paths: list of strings
    :param dest: optional destination path, which will be the first path in
                 'paths'
    :type dest: dict
    :return: path to concatenated hdf file.
    :rtype: str
    '''
    # copy hdf to temp area to build upon
    hdf_master_path = copy_file(hdf_paths[0])

    with FlightDataFile(hdf_master_path, mode='a') as fdf_master:
        fdf_master.concatenate(hdf_paths[1:])

    if dest:
        shutil.move(hdf_master_path, dest)
        return dest
    else:
        return hdf_master_path


@deprecated(details='Use FlightDataFile.trim() instead')
def strip_hdf(hdf_path, params_to_keep, dest, deidentify=True):
    '''
    Strip an HDF file of all parameters apart from those in params_to_keep.
    Does not raise an exception if any of the params_to_keep are not in the
    HDF file.

    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param params_to_keep: parameter names to keep.
    :type param_to_keep: list of str
    :param dest: destination path for stripped output file
    :type dest: str
    :return: all parameters names within the output hdf file
    :rtype: [str]
    '''
    with FlightDataFile(hdf_path) as hdf:
        hdf.trim(dest, parameter_list=params_to_keep, deidentify=deidentify)

    # XXX: filter the param_to_keep list to the list of existing parameters
    with FlightDataFile(dest) as hdf:
        return list(set(params_to_keep) & set(hdf.keys()))


def write_segment(source, segment, dest, boundary, submasks=None):
    '''
    Writes a segment of the HDF file stored in hdf_path to dest defined by
    segments, a slice in seconds. Expects the HDF file to contain whole
    superframes.

    Assumes "data" and "mask" are present.

    The source file used to be copied to the destination and then modified the
    file inplace. Since it is impossible to fully reclaim the space of deleted
    datasets, we now create a new hdf file and copy groups, attributes and
    parameters into it resulting in smaller segment sizes.

    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param segment: segment of flight to write in seconds. step is disregarded.
    :type segment: slice
    :param dest: destination path for output file containing segment.
    :type dest: str
    :param supf_boundary: Split on superframe boundaries, masking data outside of the segment.
    :type supf_boundary: bool
    :param submasks: Collection of submask names to write. The default value of None writes all submasks, while an empty collection will result in no submasks being written.
    :type submasks: collection (tuple/list/set) or None
    :return: path to output hdf file containing specified segment.
    :rtype: str

    TODO: Support segmenting parameter masks. Q: Does this mean copying the mask along
    with data? If so, this is already done.
    '''
    if os.path.isfile(dest):
        logging.warning(
            "File '%s' already exists, write_segments will delete file.", dest)
        os.remove(dest)

    supf_start_secs, supf_stop_secs, array_start_secs, array_stop_secs = segment_boundaries(segment, boundary)

    if supf_start_secs == 0 and supf_stop_secs is None:
        logging.debug("Write Segment: Segment is not being sliced, file will be copied.")
        shutil.copy(source, dest)
        return dest

    with FlightDataFile(source) as source_hdf:
        if supf_stop_secs is None:
            supf_stop_secs = source_hdf.duration

        segment_duration = supf_stop_secs - supf_start_secs

        if source_hdf.duration == segment_duration:
            logging.debug("Write Segment: Segment duration is equal to whole "
                          "duration, file will be copied.")
            shutil.copy(source, dest)
            return dest

        with FlightDataFile(dest, mode='w') as dest_hdf:
            logging.debug("Write Segment: Duration %.2fs to be written to %s",
                          segment_duration, dest)

            for group_name in source_hdf.hdf.keys():  # Copy top-level groups.
                if group_name == 'series':
                    continue  # Avoid copying parameter datasets.
                source_hdf.hdf.copy(group_name, dest_hdf.hdf)
                logging.debug("Copied group '%s' between '%s' and '%s'.",
                              group_name, source, dest)

            _copy_attrs(source_hdf.hdf, dest_hdf.hdf)  # Copy top-level attrs.

            #Q: Could this be too short if we change the start and stop a bit further down???
            dest_hdf.duration = segment_duration  # Overwrite duration.

            supf_slice = slice(supf_start_secs, supf_stop_secs)

            for param_name in source_hdf.keys():

                #Q: Why not always pad masked values to the next superframe

                param = source_hdf.get_param(
                    param_name, _slice=supf_slice, load_submasks=True)
                if ((param.hz * 64) % 1) != 0:
                    raise ValueError(
                        "Parameter '%s' does not record a consistent number of "
                        "values every superframe. Check the LFL definition."
                        % param_name)

                if submasks is not None and param.submasks:
                    # if param does not have submasks, or no submasks match,
                    # write the original mask
                    mask_subset = {k: v for k, v in param.submasks.items() if k in submasks}
                    if mask_subset and len(param.submasks) != len(mask_subset):
                        submask_arrays = list(six.itervalues(mask_subset))
                        if 'padding' in submasks and 'padding' not in param.submasks:
                            # padding submask from initial processing does not exist
                            # in this case include the original array mask
                            submask_arrays += [param.array.mask]
                        param.array.mask = merge_masks(submask_arrays)
                    param.submasks = mask_subset

                param.array = param.raw_array

                param_start_index = int(array_start_secs * param.hz)
                param_stop_index = int(len(param.array) -
                                       (array_stop_secs * param.hz))

                array_size = int(segment_duration * param.hz)
                if param.array.size < array_size:
                    # There's not enough data in the input
                    # The input data was not aligned to 4s or 64s
                    # we need to pad the arrays to have the expected number of
                    # samples
                    param_stop_index = param.array.size
                    padding_size = array_size - param_stop_index
                    param.array = np.ma.concatenate((
                        param.array,
                        np.ma.zeros(padding_size, dtype=param.array.dtype)))

                    for sub_name, submask in param.submasks.items():
                        param.submasks[sub_name] = np.ma.concatenate((
                            submask,
                            np.ma.ones(padding_size, dtype=submask.dtype)))

                # Mask data outside of split.
                param.array[:param_start_index] = np.ma.masked
                param.array[param_stop_index:] = np.ma.masked

                for submask in param.submasks.values():
                    submask[:param_start_index] = True
                    submask[param_stop_index:] = True

                # save modified parameter back to file
                dest_hdf[param_name] = param
                #logging.debug("Finished writing segment: %s", dest_hdf)

    return dest


def segment_boundaries(segment, boundary):
    '''
    Calculate start and stop boundaries from segment slice and amount of
    padding needed to fill to boundary edge
    '''
    supf_stop_secs = segment.stop

    if segment.start:
        supf_start_secs = (int(segment.start) // boundary) * boundary
        array_start_secs = segment.start % boundary
    else:
        supf_start_secs = 0
        array_start_secs = 0

    array_stop_secs = 0
    if segment.stop:
        # Always round up to next boundary
        supf_stop_secs = (int(segment.stop) // boundary) * boundary

        if segment.stop % boundary != 0:
            # Segment does not end on a frame/superframe boundary, include the
            # following frame/superframe.
            supf_stop_secs += boundary
            array_stop_secs = boundary - (segment.stop % boundary)
    return supf_start_secs, supf_stop_secs, array_start_secs, array_stop_secs


def revert_masks(hdf_path, params=None, delete_derived=False):
    '''
    :type hdf_path: str
    :type params: params to revert or delete.
    :type params: [str] or None
    :type delete_derived: bool
    '''
    with hdf_file(hdf_path) as hdf:
        if not params:
            params = hdf.keys() if delete_derived else hdf.lfl_keys()

        for param_name in params:
            param = hdf.get_param(param_name, load_submasks=True)

            if not param.lfl:
                if delete_derived:
                    del hdf[param_name]
                continue

            if 'padding' not in param.submasks:
                continue

            param.array = param.get_array(submask='padding')
            param.submasks = {'padding': param.submasks['padding']}
            param.invalid = False
            hdf[param_name] = param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command',
                                      description="Utility command, either "
                                      "'strip' or 'revert'")
    strip_parser = subparser.add_parser('strip', help='Strip a file to a '
                                        'subset of parameters.')
    strip_parser.add_argument('input_file_path', help='Input hdf filename.')
    strip_parser.add_argument('output_file_path', help='Output hdf filename.')
    strip_parser.add_argument('parameters', nargs='+',
                              help='Store this list of parameters into the '
                              'output hdf file. All other parameters will be '
                              'stripped.')
    revert_parser = subparser.add_parser('revert',
                                         help='Revert masks of parameters')
    revert_parser.add_argument('file_path', help='File path.')
    revert_parser.add_argument('-p', '--parameters', nargs='+', default=None,
                               help='Parameter names to revert.')
    revert_parser.add_argument('-d', '--delete-derived', action='store_true',
                               help='Delete derived parameters.')
    args = parser.parse_args()
    if args.command == 'strip':
        if not os.path.isfile(args.input_file_path):
            parser.error("Input file path '%s' does not exist." %
                         args.input_file_path)
        if os.path.exists(args.output_file_path):
            parser.error("Output file path '%s' already exists." %
                         args.output_file_path)
        output_parameters = strip_hdf(args.input_file_path, args.parameters,
                                      args.output_file_path)
        if output_parameters:
            print('The following parameters are in the output hdf file:')
            for name in output_parameters:
                print(' * %s' % name)
        else:
            print('No matching parameters were found in the hdf file.')
    elif args.command == 'revert':
        revert_masks(args.file_path, params=args.parameters,
                     delete_derived=args.delete_derived)
