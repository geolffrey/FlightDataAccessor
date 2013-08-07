import argparse
import logging
import math
import numpy as np
import os
import shutil

from hdfaccess.file import hdf_file

#FIXME: Reinstate import after fixed dependency on utilities
##from utilities.filesystem_tools import copy_file
def copy_file(orig_path, dest_dir=None, postfix='_copy'):
    '''
    Creates a copy of the file with the postfix inserted between the filename
    and the extension. Can create copy into a different path (will create
    folders as required)
    
    :param orig_path: Path to original file
    :type orig_path: path
    :param dest_dir: Put copy of file into a different directory, e.g. 'temp/'
    :type dest_dir: path
    :param postfix: Rename copy of file with postfix before file extension
    :type postfix: string
    '''
    assert dest_dir or postfix, "Must define either dest_dir or postfix"
    path_minus_ext, ext = os.path.splitext(orig_path)
    if dest_dir:
        filename = os.path.basename(path_minus_ext)
        path_minus_ext = os.path.join(dest_dir, filename)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    copy_path = path_minus_ext + postfix + ext
    if os.path.isfile(copy_path):
        os.remove(copy_path)
    shutil.copy(orig_path, copy_path)
    return copy_path


def _copy_attrs(source_group, target_group, deidentify=False):
    '''
    While the library can recursively copy groups and datasets, there does
    not seem to be a simple way to copy all of a group's attributes at once.
    '''
    
        
    for key, value in source_group.attrs.iteritems():
        if deidentify and key in ('aircraft_info', 'tailmark'):
            continue
        target_group.attrs[key] = value


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

    with hdf_file(hdf_master_path) as hdf_master:
        master_keys = hdf_master.keys()
        for hdf_path in hdf_paths[1:]:
            with hdf_file(hdf_path) as hdf:
                # check that all parameters match (avoids mismatching array lengths)
                param_keys = hdf.keys()
                assert set(param_keys) == set(master_keys)
                logging.debug("Copying parameters from file %s", hdf_path)
                for param_name in param_keys:
                    param = hdf[param_name]
                    master_param = hdf_master[param_name]
                    assert param.frequency == master_param.frequency
                    assert param.offset == master_param.offset
                    assert param.units == master_param.units
                    # join arrays together
                    master_param.array = np.ma.concatenate(
                        (master_param.raw_array, param.raw_array))
                    # re-save parameter
                    hdf_master[param_name] = master_param
                # extend the master's duration
                hdf_master.duration += hdf.duration
            #endwith
            logging.debug("Completed extending parameters from %s", hdf_path)
        #endfor
    #endwith

    if dest:
        shutil.move(hdf_master_path, dest)
        return dest
    else:
        return hdf_master_path


def strip_hdf(hdf_path, params_to_keep, dest, deidentify=True):
    '''
    Strip an HDF file of all parameters apart from those in params_to_keep. Does
    not raise an exception if any of the params_to_keep are not in the HDF file.

    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param params_to_keep: parameter names to keep.
    :type param_to_keep: list of str
    :param dest: destination path for stripped output file
    :type dest: str
    :return: all parameters names within the output hdf file
    :rtype: [str]
    '''
    with hdf_file(hdf_path) as hdf, hdf_file(dest, create=True) as hdf_dest:
        _copy_attrs(hdf.hdf, hdf_dest.hdf, deidentify=deidentify) # Copy top-level attrs.
        params = hdf.get_params(params_to_keep)
        for param_name, param in params.iteritems():
            hdf_dest[param_name] = param
    return params.keys()


def write_segment(source, segment, dest, supf_boundary=True):
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
    :return: path to output hdf file containing specified segment.
    :rtype: str

    TODO: Support segmenting parameter masks. Q: Does this mean copying the mask along
    with data? If so, this is already done.
    '''
    if os.path.isfile(dest):
        logging.warning(
            "File '%s' already exists, write_segments will delete file.", dest)
        os.remove(dest)

    supf_start_secs = segment.start
    supf_stop_secs = segment.stop

    boundary = 64 if supf_boundary else 4
    
    if segment.start:
        supf_start_secs = (int(segment.start) / boundary) * boundary
    else:
        supf_start_secs = 0
    if segment.stop:
        supf_stop_secs = (int(segment.stop) / boundary) * boundary
        if segment.stop % boundary != 0:
            # Segment does not end on a frame/superframe boundary, include the
            # following frame/superframe.
            supf_stop_secs += boundary

    if supf_start_secs == 0 and supf_stop_secs is None:
        logging.debug("Write Segment: Segment is not being sliced, file will be copied.")
        shutil.copy(source, dest)
        return dest

    with hdf_file(source) as source_hdf:
        if supf_stop_secs is None:
            supf_stop_secs = source_hdf.duration
        
        segment_duration = supf_stop_secs - supf_start_secs

        if source_hdf.duration == segment_duration:
            logging.debug("Write Segment: Segment duration is equal to whole "
                          "duration, file will be copied.")
            shutil.copy(source, dest)
            return dest

        with hdf_file(dest, create=True) as dest_hdf:
            logging.debug("Write Segment: Duration %.2fs to be written to %s",
                          segment_duration, dest)

            for group_name in source_hdf.hdf.keys():  # Copy top-level groups.
                if group_name == 'series':
                    continue  # Avoid copying parameter datasets.
                source_hdf.hdf.copy(group_name, dest_hdf.hdf)
                logging.debug("Copied group '%s' between '%s' and '%s'.",
                              group_name, source, dest)

            _copy_attrs(source_hdf.hdf, dest_hdf.hdf)  # Copy top-level attrs.

            dest_hdf.duration = segment_duration  # Overwrite duration.

            for param_name in source_hdf.keys():
                param = source_hdf.get_param(
                    param_name, _slice=slice(supf_start_secs, supf_stop_secs))
                if ((param.hz * 64) % 1) != 0:
                    raise ValueError(
                        "Parameter '%s' does not record a consistent number of "
                        "values every superframe. Check the LFL definition."
                        % param_name)
                if segment.start:
                    supf_start_index = int(supf_start_secs * param.hz)
                    param_start_index = int((segment.start - supf_start_secs) * param.hz)
                else:
                    supf_start_index = 0
                    param_start_index = supf_start_index
                if segment.stop:
                    supf_stop_index = int(supf_stop_secs * param.hz)
                    param_stop_index = int(segment.stop * param.hz)
                else:
                    supf_stop_index = len(param.raw_array)
                    param_stop_index = supf_stop_index

                param.array = param.raw_array
                # Mask data outside of split.
                param.array[:param_start_index] = np.ma.masked
                param.array[param_stop_index:] = np.ma.masked
                # save modified parameter back to file
                dest_hdf[param_name] = param
                #logging.debug("Finished writing segment: %s", dest_hdf)

    return dest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command',
                                      description="Utility command, currently "
                                      "only 'strip' is supported",
                                      help='Additional help')
    strip_parser = subparser.add_parser('strip')
    strip_parser.add_argument('input_file_path', help='Input hdf filename.')
    strip_parser.add_argument('output_file_path', help='Output hdf filename.')
    strip_parser.add_argument('parameters', nargs='+',
                              help='Store this list of parameters into the '
                              'output hdf file. All other parameters will be '
                              'stripped.')
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
            print 'The following parameters are in the output hdf file:'
            for name in output_parameters:
                print ' * %s' % name
        else:
            print 'No matching parameters were found in the hdf file.'
