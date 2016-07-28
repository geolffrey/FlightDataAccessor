import h5py
import argparse
import os
import sys
import json
import logging
import numpy as np

from hdfaccess.file import hdf_file
from hdfaccess.parameter import MappedArray
from analysis_engine.utils import list_parameters
from flightdatautilities import units as ut
from collections import Counter

logger = logging.getLogger(__name__)

class StoppedOnFirstError(Exception):
    pass

class HDFValidatorStreamHandler(logging.StreamHandler):
    """
    A handler to raise stop on the first error log.
    """  
    def __init__(self, stop_on_error=None):
        super(HDFValidatorStreamHandler, self).__init__()
        self.stop_on_error = stop_on_error
        self.errors = 0
        self.warnings = 0
        
    def emit(self, record):
        super(HDFValidatorStreamHandler, self).emit(record)
        if record.levelno >= 40:
            self.errors += 1
        elif record.levelname == 'WARNING':
            self.warnings += 1
        if self.stop_on_error and record.levelno >= 40:
            raise StoppedOnFirstError()
    
    def get_error_counts(self):
        return {'warnings':self.warnings, 'errors':self.errors}

VALID_FREQUENCIES = {
    # base 2 frequencies
    0.03125,
    0.015625,
    0.0625,
    0.125,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16,
    # other non base 2 frequencies
    0.016666667,
    0.05,
    0.1,
    0.2,
    5,
    10,
    20,    
}

# Extra parameters not listed from list_parameter
EXTRA_PARAMETERS = [
    u'Day',
    u'Hour',
    u'Minute',
    u'Month',
    u'Second',
    u'Year',
    u'Frame Counter',
    u'Subframe Counter',
]

CORE_PARAMETERS = [
    u'Airspeed',
    u'Heading',
    u'Altitude STD',
    u'Heading True',
]

result = {
    'passed': 0,
    'failed': 0,
    'warning': 0,
}

def subtitle(subtitle):
    title(subtitle, line='-', section=False)

def title(title, line='=', section=True):
    if section:
        logger.info("%s" % ('_'*80))
    logger.info(title)
    logger.info("%s" % (line*len(title),))

def check_parameter_names(hdf):
    subtitle("Checking parameter names")
    params_from_file = set(hdf.keys())
    parameter_naming = set(list_parameters()) | set(EXTRA_PARAMETERS)
    matched_names = set()
    for name in parameter_naming:
        found = hdf.search(name)
        if found:
            matched_names.update(found)
    unmatched_names = params_from_file - matched_names
    if not matched_names:
        logger.error(
            "All %s parameters are unrecongnised by POLARIS." \
            % (len(unmatched_names),)
        )
    elif unmatched_names:
        logger.info(
            "Number of parameter recongnised by POLARIS: %s" \
            % (len(matched_names),)
        )
        logger.warn(
            "Number of parameters unrecongnised by POLARIS: %s" \
            % (len(unmatched_names),)
        )
        logger.debug(
            "The following parameters names are unrecognised by "\
            "POLARIS: %s" % (unmatched_names,)
        )
    else:
        logger.info(
            "All %s parameters are recongnised by POLARIS." \
            % (len(matched_names),)
        )
    return (tuple(matched_names), tuple(unmatched_names))
        
def check_for_core_parameters(hdf):
    params_from_file = hdf.keys()
    airspeed = 'Airspeed' in params_from_file
    altitude = 'Altitude STD' in params_from_file
    heading = 'Heading' in params_from_file
    heading_true = 'Heading True' in params_from_file
    core_available = airspeed and altitude and (heading or heading_true)
    
    if core_available:
        logger.info("All core parameters available for analysis.")
        if heading_true and not heading:
            logger.info("Analysis will use parameter 'Heading True' as "\
                        "parameter 'Heading' not available.")
    else:
        if not airspeed:
            logger.error("Parameter 'Airspeed' not found. Required as one "\
                         "of the core parameters for analysis.")
            result['failed'] += 1
        if not altitude:
            logger.error("Parameter 'Altitude STD' not found. Required as "\
                         "one of the core parameters for analysis.")
            result['failed'] += 1
        if not heading and not heading_true:
            logger.error("Parameter 'Heading' and 'Heading True' not found. "\
                         "One of these parameters is required for analysis.")
            result['failed'] += 1
    return core_available
            
#==============================================================================
#   Parameter's Attributes
#==============================================================================
def validate_parameters(hdf):
    title("Checking Parameters")
    rc = 0
    matched, unmatched = check_parameter_names(hdf)
    core_available = check_for_core_parameters(hdf)    
    for name, parameter in hdf.iteritems():
        title("Checking Parameter: %s" % (name,))
        if name in matched:
            logger.info("'%s' is a recongnised by POLARIS." % (name,))
        else:
            logger.warn("'%s' is a unrecongnised by POLARIS." % (name,))
        if name in CORE_PARAMETERS:
            logger.info("'%s' is a core parameter required for basic "\
                        "analysis." % (name,))
        validate_parameter_attributes(hdf, name, parameter)
        validate_parameters_dataset(hdf, name, parameter)
    return rc

def validate_parameter_attributes(hdf, name, parameter):
    subtitle("Checking Attribute for Parameter: %s" % (name,))
    validate_arinc_429(hdf, name, parameter)
    validate_data_type(hdf, name, parameter)
    validate_frequency(hdf, name, parameter)
    validate_lfl(hdf, name, parameter)
    validate_name(hdf, name, parameter)
    validate_source_name(hdf, name, parameter)
    validate_supf_offset(hdf, name, parameter)
    validate_units(hdf, name, parameter)
    validate_values_mapping(hdf, name, parameter)

#==============================================================================
#   Parameter's Attributes
#==============================================================================
def validate_parameters_dataset(hdf, name, parameter):
    subtitle("Checking dataset for Parameter: %s" % (name,))
    validate_dataset(hdf, name, parameter)

def validate_arinc_429(hdf, name, parameter):
    logger.info("Checking parameter attribute: arinc_429")
    if parameter.arinc_429 is None:
        logger.warn("No attribute 'arinc_429'. '%s' does not have an ARINC "\
                    "429 source." % name)
    else:
        if 'bool' not in type(parameter.arinc_429).__name__:
            logger.error("Error: Attribute 'arinc_429' is not a Boolean type.")
            result['failed'] += 1
        if parameter.arinc_429:
            logger.info("'%s' has an ARINC 429 source." % name)
        else:
            logger.info("'%s' does not have an ARINC 429 source." % name)

def validate_data_type(hdf, name, parameter):
    logger.info("Checking parameter attribute: data_type")
    if parameter.data_type is None:
        logger.error(
            "No attribute 'data_type' present for '%s'. "\
            "This is required attribute." % (name,)
        )
        result['failed'] += 1 
    else:
        logger.info("'%s' has a 'data_type' attribute of: %s" \
                    % (name, parameter.data_type))
        logger.info("'%s' data has a dtype of: %s" \
                    % (name, parameter.array.data.dtype))
        if parameter.data_type in ['ASCII',]:
            if 'string' not in parameter.array.dtype.name:
                logger.error(
                    "'%s' data type is %s. It should be a string for '%s' "\
                    "parameters. " % (name, parameter.array.dtype.name,
                                      parameter.data_type)                     
                )
                result['failed'] += 1
                return
        elif parameter.data_type in ['BCD', 'Interpolated', 'Polynomial',
                                     'Signed', 'Synchro', 'Unsigned']:
            if 'float' not in parameter.array.dtype.name:
                logger.error(
                    "'%s' data type is %s. It should be a float for '%s' "\
                    "parameters. " % (name, parameter.array.dtype.name,
                                      parameter.data_type)                    
                )
                result['failed'] += 1
                return
        elif parameter.data_type in ['Multi-state', 'Discrete']:
            if 'int' not in parameter.array.dtype.name:
                logger.error(
                    "'%s' data type is %s. It should be an integer for '%s' "\
                    "parameters. " % (name, parameter.array.dtype.name,
                                      parameter.data_type)
                )
                result['failed'] += 1
                return 
        logger.info(
            "'%s' data_type is %s and is an array of %s." \
            % (name, parameter.data_type, parameter.array.dtype.name)
        )
   

def validate_frequency(hdf, name, parameter):
    logger.info("Checking parameter attribute: frequency")
    if parameter.frequency is None:
        logger.error("No attribute 'frequency' present for '%s'. "\
                     "This is required attribute." % (name,))
        result['failed'] += 1
    else:
        if parameter.frequency not in VALID_FREQUENCIES:
            logger.error(
                "'%s' has a 'frequency' of %s which is not a " \
                "frequency supported by POLARIS." \
                % (name, parameter.frequency)
            )
            result['failed'] += 1
        else:
            logger.info(
                "'frequency' is %s Hz for '%s' and is a support frequency." \
                % (parameter.frequency, name)
            )
        if hdf.frequencies is not None:
            if 'array' in type(hdf.frequencies).__name__:
                if parameter.frequency not in hdf.frequencies:
                    logger.warn("Frequency not in the Root attribute list "\
                                "of frequenices.")
            elif parameter.frequency != hdf.frequencies:
                logger.warn("Frequency not in the Root attribute list "\
                            "of frequenices.")

def validate_lfl(hdf, name, parameter):
    ''' 
    Check that the required lfl attribute is present. Report if recored or 
    derived.
    '''
    logger.info("Checking parameter attribute: lfl")
    if parameter.lfl is None:
        logger.error("Error: No attribute 'lfl' for '%s'. Attribute "\
                     "is Required." % (name,))
        result['failed'] += 1
        return
    if 'bool' not in type(parameter.lfl).__name__:
        logger.error("Error: lfl should be an Boolean. Type is %s" \
                     % (type(parameter.lfl).__name__,))
        result['failed'] += 1
    if parameter.lfl:
        logger.info("'%s' is a recorded parameter." % (name,))
    else:
        logger.info("'%s' is a derived parameter." % (name,))

def validate_name(hdf, name, parameter):
    logger.info("Checking parameter attribute: name")
    if parameter.name is None:
        logger.error("No attribute 'name' for '%s'. Attribute is Required."\
                     % (name,))
        result['failed'] += 1
    else:
        if parameter.name != name:
            logger.error(
                "Error: 'name' is present, but is not the same name as "\
                "the parameter group. name: %s, parameter group: %s" \
                % (parameter.name, name)
            )
            result['failed'] += 1
        else:
            logger.info("'name' is present and name is the same name as "\
                  "the parameter group.")


def validate_source_name(hdf, name, parameter):
    logger.info("Checking parameter attribute: source_name")
    if parameter.source_name is None:
        logger.info(
            "No attribute 'source_name' for '%s'. Attribute is optional. "\
            % (name,)
        )
    else:
        logger.info("'source_name' is present. Original name %s maps to "\
              "POLARIS name %s" % (parameter.source_name, name))

def validate_supf_offset(hdf, name, parameter):
    logger.info("Checking parameter attribute: supf_offset")
    if parameter.offset is None:
        logger.error("No attribute 'supf_offset' for '%s'. "\
                     "Attribute is Required. " % (name,))
        result['failed'] += 1
    else:
        if 'float' not in type(parameter.offset).__name__:
            logger.error(
                "'supf_offset' type for '%s' is not a float. Got %s instead" \
                % (name, type(parameter.offset).__name__)
            )
            result['failed'] += 1
        else:
            logger.info("'supf_offset' is present and correct data type.")
    

def validate_units(hdf, name, parameter):
    logger.info("Checking parameter attribute: units")

    if parameter.units is None:
        logger.error("No attribute 'units' for '%s'. Attribute is Required."\
                     % (name,))
        result['failed'] += 1
    else:
        if type(parameter.units).__name__ not in ['str', 'string', 'string_']:
            logger.error("'units' expected to be a string, got %s" \
                         % (type(parameter.units).__name__))    
        if parameter.units == '':
            logger.info("Attribute 'units' is present for '%s', but empty."\
                        % (name,))
        elif parameter.units in ut.available():
            logger.info(
                "Attribute 'units' is present for '%s' and has a valid unit "\
                "of '%s'." % (name, parameter.units)
            )   
        else:
            logger.error("Attribute 'units' is present for '%s' and has an "\
                         "unknown unit of '%s'." % (name, parameter.units))
            result['failed'] += 1


def validate_values_mapping(hdf, name, parameter):
    logger.info("Checking parameter attribute: values_mapping")
    if parameter.values_mapping is None:
        if parameter.data_type in ('Discrete', 'Multi-state',
                                   'Enumerated Discrete'):    
            logger.error("No attribute 'values_mapping' for '%s'. "\
                         "Attribute is Required for a $s parameter. "\
                         % (name, parameter.data_type))
            result['failed'] += 1
        else:
            logger.info("No attribute 'values_mapping' not required for '%s'."\
                        % (name, ))
    else:
        logger.info("Attribute, 'values_mapping' value is: %s"\
                    % (parameter.values_mapping,))
        try:
            # validate JSON string
            jstr = json.loads(
                hdf.hdf['/series/' + name].attrs['values_mapping']
            )
            logger.info("Attribute, 'values_mapping' is a valid json "\
                        "string: %s" % (jstr,))
        except ValueError as e:
            logger.error("'values_mapping' is not a valid JSON string. (%s)"\
                         % (e, ))
            result['failed'] += 1
        if parameter.data_type == 'Discrete':
            try: 
                value0 = parameter.values_mapping[0] # False values
                logger.debug("discrete value of 0 maps to '%s'." % (value0))
            except KeyError as e:
                logger.debug("discrete value of 0 has no mapping.")
            try:
                value1 = parameter.values_mapping[1] # True values
                if value1 in ["", "-"]:
                    logger.error("discrete value of 1 should map to a non "\
                                 "emtpy (or '-') string value of 1. Got '%s'." \
                                 % (value1,))
                else:
                    logger.debug("discrete value of 1 maps to '%s'" % (value1))                
            except:
                logger.error("discrete value of 1 has no mapping. Needs "\
                             "to have a mapping for this value.")
            if len(parameter.values_mapping.keys()) > 2:
                logger.error("'%s' a discrete parameter, but values_mapping "\
                             "attribute has %s values should be no "\
                             "more than 2." \
                             % (name, len(parameter.data_type.keys())))
                


def validate_dataset(hdf, name, parameter):
    logger.info("Checking parameter dataset for inf and NaN values.")
    if 'int' in parameter.array.dtype.name or \
       'float' in parameter.array.dtype.name:

        nan_unmasked = np.ma.masked_equal(
            np.isnan(parameter.array),False).count()
        nan_count = np.ma.masked_equal(
            np.isnan(parameter.array.data),False).count()
        inf_unmasked = np.ma.masked_equal(
            np.isinf(parameter.array),False).count()
        inf_count = np.ma.masked_equal(
            np.isinf(parameter.array.data),False).count()        
        
        if nan_count != 0:
            logger.error("%s NaN values found in the data of '%s'." \
                         % (nan_count, name))
            logger.info("NaN values not masked: %s" % (nan_unmasked))
            result['failed'] += 1
        if inf_count != 0:
            logger.error("%s inf values found in the data of '%s'." \
                         % (nan_count, name))
            logger.info("inf values not masked: %s" % (inf_unmasked))
            result['failed'] += 1
        if nan_count == inf_count == 0:
            logger.info("dataset does not have any inf or NaN values.")
    
    logger.info("Checking parameter actual dataset size against, "\
                "expected size of duration * param_freqr.")    
    expected_array_size = hdf.duration * parameter.frequency
    actual_data_size = len(parameter.array)
    if expected_array_size != actual_data_size:
        logger.error("The data size of '%s' different to expected size of %s."\
                     % (actual_data_size, expected_array_size))
        result['failed'] += 1
    else:
        logger.info("Data array is of the expected size of %s." \
                    % (expected_array_size,))
    if len(parameter.array.data) != len(parameter.array.mask):
        logger.error("The data and mask sizes of '%s' are different.")
        result['failed'] += 1
    else:
        logger.info("Data and Mask both have the size of %s elements." \
                    % (len(parameter.array.data)))

    logger.info("Checking dataset type and shape.")
    isMaskedArray = isinstance(parameter.array,np.ma.core.MaskedArray)
    isMappedArray = isinstance(parameter.array,MappedArray)
    if not isMaskedArray and not isMappedArray:
        logger.error("Data for %s is not a MaskedArray or MappedArray. "\
                     "Type is %s" % (name,type(parameter.array)),)
    else:
        # check shape, it should be 1 dimensional arrays for data and mask
        if len(parameter.array.shape) != 1:
            logger.error("The data and mask are not in an 1 dimensional "\
                         "array. The data's shape is %s " \
                         % (parameter.array.shape,))
        else:
            logger.info("Data is in a %s with a shape of %s"\
                         % (type(parameter.array).__name__,
                            parameter.array.shape, ))
        
def validate_namespace(hdf5):
    '''Uses h5py functions to verify what is stored on the root group'''
    found = ''
    title("Checking for the namespace 'series' group on root")
    if 'series' in hdf5.keys():
        logger.info("Found the POLARIS namespace 'series' on root.")
        found = 'series'
        result['passed'] += 1
    else:
        found = [g for g in hdf5.keys() if 'series' in g.lower()]
        if found:
            # series found but in the wrong case.
            logger.error("Namespace '%s' found, but needs to be in "\
                         "lower case." % (found,))
            result['failed'] += 1
        else:
            logger.error("Namespace 'series' was not found on root.")
            result['failed'] += 1
                
    logger.info("Checking for other namespace groups on root.")
    group_num = len(hdf5.keys())
    
    show_groups = False
    if group_num is 1 and 'series' in found:
        logger.info("Namespace 'series' is the only group on root.")
        result['passed'] += 1
    elif group_num is 1 and 'series' not in found:
        logger.error("Only one namespace on root, but it's not 'series'.")
        result['failed'] += 1
        show_groups = True
    elif group_num is 0:
        logger.error("No namespace groups in found in the file.")
        result['failed'] += 1
    elif group_num > 1 and 'series' in found:
        logger.warn("Namespace 'series' found along with %s addtional "\
                    "groups." % (group_num-1,))
        result['warning'] += 1
        show_groups = True
    elif group_num > 1:
        logger.error("%s namespace groups are on root, but not 'series'." \
                     % (group_num,))
        result['failed'] += 1
        show_groups = True
    if show_groups:
        logger.debug("The following namespace groups are on root: %s" \
                     % ([g for g in hdf5.keys() if 'series' not in g],))
        logger.info("If these are parmeters they must be stored "\
                    "within 'series'.")
        
#==============================================================================
#   Root Attributes
#==============================================================================
def validate_root_attribute(hdf):
    title("Checking the Root attributes")
    validate_duration_attribute(hdf)
    validate_frequencies_attribute(hdf)
    validate_reliable_frame_counter_attribute(hdf)
    validate_reliable_subframe_counter_attribute(hdf)
    validate_start_timestamp_attribute(hdf)
    validate_superframe_present_attribute(hdf)


def validate_duration_attribute(hdf):
    logger.info("Checking Root Attribute: duration")
    if hdf.duration:
        logger.info("duration attribute present with a value of %s." \
              % (hdf.duration,))
        result['passed'] += 1
        if 'int' in type(hdf.hdf.attrs['duration']).__name__:
            logger.info("Passed: duration is an integer.")
            result['passed'] += 1
        else:
            logger.error("duration is not an integer, type reported as '%s'." \
                         % (type(hdf.hdf.attrs['duration']).__name__,))
            result['failed'] += 1        
    else:
        logger.error("No root attribrute 'duration'. This is a required "\
                     "attribute.")
        result['failed'] += 1

def validate_frequencies_attribute(hdf):
    logger.info("Checking Root Attribute: frequencies")
    if hdf.frequencies is not None:
        logger.info("frequencies attribute present.")
        name = type(hdf.frequencies).__name__
        if 'array' in name or 'list' in name:
            floatcount = 0
            rf = set(list(hdf.frequencies))
            for value in hdf.frequencies:
                if 'float' in type(value).__name__:
                    floatcount += 1
                else:
                    logger.error("frequency value %s should be a float" \
                                 % (value))
                    result['failed'] += 1
            if floatcount == len(hdf.frequencies):
                logger.info("Passed: frequencies listed are float values.")
            else:
                logger.error("Some or all frequencies are not float values.")
                result['failed'] += 1
        elif 'float' in name:
            logger.info("Passed: frequency listed is a float value.")
            rf = set([hdf.frequencies])
        else:
            logger.error("Error: frequency listed is not a float value.")
            result['failed'] += 1
            
        
        pf = set([v.frequency for k,v in hdf.iteritems()])
        if rf == pf:
            logger.info("Root frequency list covers all the frequencies "\
                        "used by parameters.")
        elif rf - pf:
            logger.info("Root frequencies lists has more frequencies than "\
                        "used by parameters. Unused frquencies: %s" \
                        % (list(rf - pf),))
        elif pf - rf:
            logger.info("More parameter frequencies used than listed in root "\
                        "attribute frequencies. Frequency not listed: %s" \
                        % (list(pf - rf),))
    else:
        logger.info("frequencies attribute not present and is optional.")

def validate_reliable_frame_counter(hdf):
    try:
        fc = hdf['Frame Counter']
    except:
        return False
    if np.ma.masked_inside(fc, 0,4095).count() != 0:
        return False
    # from split_hdf_to_segments.py
    fc_diff = np.ma.diff(fc.array)
    fc_diff = np.ma.masked_equal(fc_diff, 1)
    fc_diff = np.ma.masked_equal(fc_diff, -4095)  
    if dfc_diff.count() == 0:
        return True 
    return False

def validate_reliable_frame_counter_attribute(hdf):
    logger.info("Checking Root Attribute: 'reliable_frame_counter'")
    parameter_exists = 'Frame Counter' in hdf.keys()
    reliable = validate_reliable_frame_counter(hdf)
    attribute_value = hdf.reliable_frame_counter
    correct_type = type(hdf.reliable_frame_counter) == bool
    if attribute_value is None:
        logger.error("'reliable_frame_counter' attribute not present "\
                     "and is required.")
        result['failed'] += 1
    else:
        logger.info("'reliable_frame_counter' attribute is present.")
        if parameter_exists and reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and is "\
                        "reliable and 'reliable_frame_counter' marked "\
                        "as reliable.")
        elif parameter_exists and reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and is "\
                        "reliable, but 'reliable_frame_counter' not "\
                        "marked as reliable.")
        elif parameter_exists and not reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and not "\
                        "reliable, but 'reliable_frame_counter' is marked "\
                        "as reliable.")
        elif parameter_exists and not reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and not "\
                        "reliable and 'reliable_frame_counter' not marked "\
                        "as reliable.")
        elif not parameter_exists and attribute_value:
            logger.info("'Frame Counter' parameter not present, but "\
                        "'reliable_frame_counter' marked as reliable. "\
                        "Value should be False.")
        elif not parameter_exists and not attribute_value:
            logger.info("'Frame Counter' parameter not present and "\
                        "'reliable_frame_counter' correctly set to False.")

def validate_reliable_subframe_counter(hdf):
    try:
        sfc = hdf['Subframe Counter']
    except:
        return False
    sfc_diff = np.ma.masked_equal(np.ma.diff(sfc.array), 1)
    if sfc_diff.count() == 0:
        return True 
    return False

def validate_reliable_subframe_counter_attribute(hdf):
    logger.info("Checking Root Attribute: 'reliable_subframe_counter'")
    parameter_exists = 'Subframe Counter' in hdf.keys()
    reliable = validate_reliable_subframe_counter(hdf)
    attribute_value = hdf.reliable_subframe_counter
    correct_type = type(hdf.reliable_subframe_counter) == bool
    if attribute_value is None:
        logger.error("'reliable_subframe_counter' attribute not present "\
                     "and is required.")
        result['failed'] += 1
    else:
        logger.info("'reliable_subframe_counter' attribute is present.")
        if parameter_exists and reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and is "\
                        "reliable and 'reliable_subframe_counter' marked "\
                        "as reliable.")
        elif parameter_exists and reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and is "\
                        "reliable, but 'reliable_subframe_counter' not "\
                        "marked as reliable.")
        elif parameter_exists and not reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and not "\
                        "reliable, but 'reliable_subframe_counter' is "\
                        "marked as reliable.")
        elif parameter_exists and not reliable and not attribute_value:
            logger.iofo("'Frame Counter' parameter is present and not "\
                        "reliable and 'reliable_subframe_counter' not "\
                        "marked as reliable.")
        elif not parameter_exists and attribute_value:
            logger.info("'Frame Counter' parameter not present, but "\
                        "'reliable_subframe_counter' marked as reliable. "\
                        "Value should be False.")
        elif not parameter_exists and not attribute_value:
            logger.info("'Frame Counter' parameter not present and "\
                        "'reliable_subframe_counter' correctly set to False.")

def validate_start_timestamp_attribute(hdf):
    logger.info("Checking Root Attribute: start_timestamp")
    if hdf.start_datetime:
        logger.info("start_timestamp attribute present.")
        logger.info("Time reported to be, %s" % (hdf.start_datetime))
        logger.info("Epoch timestamp value is: %s" \
                    % (hdf.hdf.attrs['start_timestamp'],))
        if 'float' in type(hdf.hdf.attrs['start_timestamp']).__name__:
            logger.info("start_timestamp is a float.")
        else:
            logger.error("Error: start_timestamp is not a float, type "\
                         "reported as '%s'." \
                         % (type(hdf.hdf.attrs['start_timestamp']).__name__,))
            result['failed'] += 1
    else:
        logger.info("'start_timestamp' attribute not present and is optional.")

def validate_superframe_present_attribute(hdf):
    logger.info("Checking Root Attribute: superframe_present.")
    if hdf.superframe_present:
        logger.info("superframe_present attribute present.")
    else:
        logger.info("superframe_present attribute is not present and "\
                    "is optional.")


def validate_file(hdffile):
    filename = hdffile.split(os.sep)[-1]
    open_with_h5py = False
    logger.info("Verifying file '%s' with FlightDataAccessor." % (filename,))
    try:
        hdf = hdf_file(hdffile, read_only=True)
        result['passed'] += 1
    except Exception as e:
        logger.error("FlightDataAccessor cannot open '%s'. "\
                     "Exception(%s: %s)" % (filename, type(e).__name__, e))

        result['failed'] += 1
        logger.info("Checking that H5PY package can read the file.")
        open_with_h5py = True
    # If FlightDataAccessor errors upon opening it maybe because '/series'
    # is not included in the file. hdf_file attempts to create and fails 
    # because we opening it as readonly. Verify the group structure by 
    # using H5PY
    if open_with_h5py:
        try:
            hdf = h5py.File(hdffile, 'r')
            logger.info("File %s can be opened by H5PY, suggesting "\
                        "the format is not compatible for POLARIS to use." \
                        % (filename,))
        except Exception as e:
            logger.error("cannot open '%s' using H5PY. Exception(%s: %s)" \
                         % (filename, type(e).__name__, e))
            return
        validate_namespace(hdf)
    else:
        validate_namespace(hdf.hdf)
        #continue testing using FlightDataAccessor
        validate_root_attribute(hdf)
        validate_parameters(hdf)
    hdf.close()

def main():
    parser = argparse.ArgumentParser(
        description="Flight Data Services, HDF5 Validator for POLARIS "\
        "compatibility",
        version="1.0"
    )
    parser.add_argument(
        "-s",
        "--stop",
        help="stop on the first error encountered.",
        action="store_true"
    )
    parser.add_argument(
        "-n",
        "--nolog",
        help="no log report file generated.",
        action="store_true"
    ) 
    parser.add_argument(
        "-e",
        "--erroronly",
        help="display only warnings and errors on screen.",
        action="store_true"
    )     
    parser.add_argument(
        'file',
        help="Input HDF5 to be tested for POLARIS compatibility."
    )    
    args = parser.parse_args()

    #Setup logger 
    #fmtr = logging.Formatter(r'%(levelname)-8s - %(name)-8s - %(message)s')
    fmtr = logging.Formatter(r'%(levelname)-9s:%(message)s')
    #setup a file handler 
    if args.nolog is False:
        logfilename = args.file.split(os.sep)[-1]
        logfilename = logfilename.replace('.hdf5','')\
            .replace('.bz2','').replace('.gz','')
        logfilename = logfilename + ".log"
        handler = logging.FileHandler(logfilename, mode='w')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmtr)
        logger.addHandler(handler)
    
    #setup a stream handler
    sh = HDFValidatorStreamHandler(args.stop)
    error_count = sh.get_error_counts()

    if args.erroronly:
        sh.setLevel(logging.ERROR)
    else:
        sh.setLevel(logging.INFO)
    sh.setFormatter(fmtr)
    
    logger.addHandler(sh)
    
    logger.setLevel(logging.DEBUG)
    logger.debug("Arguments: %s" % (str(args),))
    try:
        validate_file(args.file)
    except StoppedOnFirstError:
        logger.info("First error encountered. Stopping as requested.")
    
    for hdr in logger.handlers:    
        if isinstance(hdr, HDFValidatorStreamHandler):
            error_count = hdr.get_error_counts()

    title("Results")
    logger.info("Validation ended with, %s errors and %s warnings" \
                % (error_count['errors'], error_count['warnings']))

if __name__ == '__main__':
    main()