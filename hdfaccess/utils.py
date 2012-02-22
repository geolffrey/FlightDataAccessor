import logging
import math
import shutil
import numpy as np
import h5py

from hdfaccess.file import hdf_file

from utilities.filesystem_tools import copy_file

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
                        (master_param.array, param.array))
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


def strip_hdf(hdf_path, params_to_keep, dest):
    '''
    Strip an HDF file of all parameters apart from those in param_names. Does
    not raise an exception if any of the params_to_keep are not in the HDF file.
    
    :param hdf_path: file path of hdf file.
    :type hdf_path: str
    :param params_to_keep: parameter names to keep.
    :type param_to_keep: list of str
    :param dest: destination path for stripped output file
    :type dest: str
    :return: path to output hdf file containing specified segment.
    :rtype: str
    '''
    # Q: Is there a better way to clone the contents of an hdf file?
    shutil.copy(hdf_path, dest)
    with h5py.File(hdf_path, 'r+') as hdf:
        for param_name in hdf['series'].keys():
            if param_name not in params_to_keep:
                del hdf['series'][param_name]
    return dest


def write_segment(hdf_path, segment, dest, supf_boundary=True):
    '''
    Writes a segment of the HDF file stored in hdf_path to dest defined by 
    segments, a slice in seconds. Expects the HDF file to contain whole
    superframes.
    
    Assumes "data" and "mask" are present.
    
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
    
    TODO: Support segmenting parameter masks
    '''
    # Q: Is there a better way to clone the contents of an hdf file?
    shutil.copy(hdf_path, dest)
    param_name_to_array = {}
    
    supf_start_secs = segment.start
    supf_stop_secs = segment.stop
    
    if supf_boundary:
        if segment.start:
            supf_start_secs = (int(segment.start) / 64) * 64
            param_start_secs = (segment.start - supf_start_secs)
        if segment.stop:
            supf_stop_secs = ((int(segment.stop) / 64) * 64)
            if segment.stop % 64 != 0:
                # Segment does not end on a superframe boundary, include the 
                # following superframe.
                supf_stop_secs += 64
                
    if supf_start_secs is None and supf_stop_secs is None:
        logging.debug("Segment is not being sliced, nothing to do")
        return dest
                
    with hdf_file(dest) as hdf:
        if supf_start_secs is None:
            segment_duration = supf_stop_secs
        elif supf_stop_secs is None:
            segment_duration = hdf.duration - supf_start_secs
        else:
            segment_duration = supf_stop_secs - supf_start_secs
        
        if hdf.duration == segment_duration:
            logging.debug("Segment duration is equal to whole duration, nothing to do")
            return dest
        else:
            hdf.duration = segment_duration
            
        for param_name in hdf.keys():
            param = hdf[param_name]
            if supf_boundary:
                if ((param.hz * 64) % 1) != 0:
                    raise ValueError("Parameter '%s' does not record a consistent "
                                     "number of values every superframe. Check the "
                                     "LFL definition." % param_name)
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
                    supf_stop_index = len(param.array)
                    param_stop_index = supf_stop_index
            
                param.array = param.array[supf_start_index:supf_stop_index]
                # Mask data outside of split.
                param.array[:param_start_index] = np.ma.masked
                param.array[param_stop_index:] = np.ma.masked
            else:
                start = int(segment.start * param.hz) if segment.start else 0
                stop = int(math.ceil(segment.stop * param.hz)) if segment.stop else len(param.array)
                param.array = param.array[start:stop]
            # save modified parameter back to file
            hdf[param_name] = param
    
    return dest