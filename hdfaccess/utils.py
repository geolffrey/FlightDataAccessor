import math
import shutil
import numpy as np
import h5py

from hdfaccess.file import hdf_file

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
    param_name_to_arrays = {}
    for hdf_path in hdf_paths:
        with h5py.File(hdf_path, 'r') as hdf:
            for param_name, param_group in hdf['series'].iteritems():
                try:
                    param_name_to_arrays[param_name].append(param_group['data'][:])
                except KeyError:
                    param_name_to_arrays[param_name] = [param_group['data'][:]]
    if dest:
        # Copy first file in hdf_paths so that the concatenated file includes
        # non-series data. XXX: Is there a simple way to do this with h5py?
        shutil.copy(hdf_paths[0], dest)
        
    else:
        dest = hdf_paths[0]
    with h5py.File(dest, 'r+') as dest_hdf:
        for param_name, array_list in param_name_to_arrays.iteritems():
            concat_array = np.concatenate(array_list)
            param_group = dest_hdf['series'][param_name]
            del param_group['data']
            param_group.create_dataset("data", data=concat_array, maxshape=(len(concat_array),))
    return dest


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
    duration = None
    
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
            param_stop_secs = (supf_stop_secs - segment.stop)
        
    #with h5py.File(hdf_path, 'r') as hdf:
    with hdf_file(dest) as hdf:
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
            
                param.aray = param.array[supf_start_index:supf_stop_index]
                # Mask data outside of split.
                param.array[:param_start_index] = np.ma.masked
                param.array[param_stop_index:] = np.ma.masked
            else:
                start = int(segment.start * param.hz) if segment.start else 0
                stop = int(math.ceil(segment.stop * param.hz)) if segment.stop else len(param.array)
                param.array = param.array[start:stop]
            # save modified parameter back to file
            hdf[param_name] = param
        
            if not duration and param.hz == 1:
                # Source duration from a 1Hz parameter.
                duration = len(segment_data)
    
    with h5py.File(dest, 'r+') as hdf:
        for param_name, arrays in param_name_to_array.iteritems():
            data, mask = arrays
            param_group = hdf['series'][param_name]
            del param_group['data']
            param_group.create_dataset("data", data=data,
                                       maxshape=(len(data),))
            del param_group['mask']
            param_group.create_dataset("mask", data=mask,
                                       maxshape=(len(mask),))
        hdf.attrs['duration'] = duration
    
    return dest