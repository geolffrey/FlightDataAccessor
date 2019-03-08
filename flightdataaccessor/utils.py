from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import shutil
import tempfile
import warnings

import six
from deprecation import deprecated

import flightdataaccessor


@deprecated(details='Use FlightDataFormat.concatenate() instead')
def concat_hdf(sources, dest=None):
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
    target = dest if dest is not None else None
    if isinstance(sources[0], six.string_types):
        # the frst source needs to be upgraded first
        if target is None:
            f, target = tempfile.mkstemp()
            os.fdopen(f, 'w').close()
            os.unlink(target)

        with flightdataaccessor.open(sources[0]) as fdf_master:
            fdf_master.upgrade(target)

        if dest is None:
            # the first source is the concatenation target
            shutil.copy(target, sources[0])

    with flightdataaccessor.open(target, mode='a') as fdf:
        fdf.concatenate(sources[1:])

    return target


@deprecated(details='Use FlightDataFormat.trim() instead')
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
    with flightdataaccessor.open(hdf_path) as fdf:
        fdf.trim(dest, parameter_list=params_to_keep, deidentify=deidentify)

    # XXX: filter the param_to_keep list to the list of existing parameters
    with flightdataaccessor.open(dest) as fdf:
        return list(set(params_to_keep) & set(fdf.keys()))


@deprecated(details='Use FlightDataFormat.trim() instead')
def write_segment(source, segment, part=0, dest=None, dest_dir=None, boundary=4, submasks=None):
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
    :param submasks: Collection of submask names to write. The default value of None writes all submasks, while an empty
        collection will result in no submasks being written.
    :type submasks: collection (tuple/list/set) or None
    :return: path to output hdf file containing specified segment.
    :rtype: str

    TODO: Support segmenting parameter masks. Q: Does this mean copying the mask along
    with data? If so, this is already done.
    '''
    # XXX: handle source / dest logic somewhere else
    if dest is None:
        if isinstance(source, six.string_types):
            source_path = source
        elif hasattr(source, 'path'):
            source_path = source.path
        else:
            source_path = None

        if dest_dir is None:
            dest_dir = os.path.dirname(dest) if isinstance(dest, str) else None

        if source_path:
            # write segment to new split file (.001)
            if dest_dir is None:
                dest_dir = os.path.dirname(source_path)
            basename = os.path.basename(source_path)
            dest_basename = os.path.splitext(basename)[0] + '.%03d.hdf5' % part
            dest = os.path.join(dest_dir, dest_basename)

    if isinstance(dest, six.string_types):
        if os.path.isfile(dest):
            logging.warning("File '%s' already exists, write_segment will delete the file.", dest)
            os.remove(dest)

    if submasks:
        warnings.warn(
            'Selection of submasks was requested which is not supported. All submasks will be saved instead',
            DeprecationWarning)

    with flightdataaccessor.open(source) as fdf:
        if not fdf.superframe_present and boundary not in (1, 4):
            # boundary in subframes
            warnings.warn(
                'Alignment to %d subframes was requested. Alignment to 64 subframes is supported only for data '
                'with superframes otherwise alignment to 4 subframes is used. Default alignment will be used instead.'
                % boundary, DeprecationWarning)
        return fdf.trim(dest, start_offset=segment.start, stop_offset=segment.stop, superframe_boundary=boundary != 1)


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
    with flightdataaccessor.open(hdf_path, mode='a') as fdf:
        if not params:
            params = fdf.keys() if delete_derived else fdf.lfl_keys()

        for param_name in params:
            param = fdf.get_param(param_name, load_submasks=True)

            if not param.lfl:
                if delete_derived:
                    del fdf[param_name]
                continue

            if 'padding' not in param.submasks:
                continue

            param.array = param.get_array(submask='padding')
            param.submasks = {'padding': param.submasks['padding']}
            param.invalid = False
            fdf[param_name] = param


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
