#!/usr/bin/env python
'''
HDFValidator checks flight data, stored in a HDF5 file format, is in a
compatible structure meeting POLARIS pre-analysis specification.
'''
import argparse
import os
import json
import logging
import h5py
import numpy as np
from math import ceil

from hdfaccess.file import hdf_file
from hdfaccess.parameter import MappedArray
from hdfaccess.tools.parameter_lists import PARAMETERS_FROM_FILES
from analysis_engine.utils import list_parameters
from flightdatautilities.patterns import wildcard_match, WILDCARD
from flightdatautilities import units as ut

logger = logging.getLogger(__name__)


class StoppedOnFirstError(Exception):
    """ Exception class used if user wants to stop upon the first error."""
    pass


class HDFValidatorHandler(logging.Handler):
    """
    A handler to raise stop on the first error log.
    """
    def __init__(self, stop_on_error=None):
        super(HDFValidatorHandler, self).__init__()
        self.stop_on_error = stop_on_error
        self.errors = 0
        self.warnings = 0

    def emit(self, record):
        ''' Log message. Then increment counter if message is a warning or
            error. Error count includes critical errors. '''
        #super(HDFValidatorStreamHandler, self).emit(record)
        if record.levelno >= logging.ERROR:
            self.errors += 1
        elif record.levelno == logging.WARN:
            self.warnings += 1
        if self.stop_on_error and record.levelno >= logging.ERROR:
            raise StoppedOnFirstError()

    def get_error_counts(self):
        ''' returns the number of warnings and errors logged.'''
        return {'warnings': self.warnings, 'errors': self.errors}

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

# -----------------------------------------------------------------------------
# Collection of parameters known to Polaris
# -----------------------------------------------------------------------------

# Main list of parameters that from the Polaris analysis engine
PARAMETERS_ANALYSIS = list_parameters()

# Minimum list of parameters (including alternatives) needed in the HDF file.
# See check_for_core_parameters method
PARAMETERS_CORE = [
    u'Airspeed',
    u'Heading',
    u'Altitude STD',
    # Helicopter only
    u'Nr',
    # Alternatives
    u'Heading True',
    u'Nr (1)',
    u'Nr (2)',
]

# Extra parameters not listed from list_parameter
PARAMETERS_EXTRA = [
    u'Day',
    u'Hour',
    u'Minute',
    u'Month',
    u'Second',
    u'Year',
    u'Frame Counter',
    u'Subframe Counter',
]

PARAMETER_LIST = list(set(PARAMETERS_FROM_FILES + PARAMETERS_ANALYSIS +
                          PARAMETERS_CORE + PARAMETERS_EXTRA))
# -----------------------------------------------------------------------------


def log_title(title, line='=', section=True):
    """Add visual breaks in the logging for main sections."""
    if section:
        logger.info("%s", '_'*80)
    logger.info(title)
    logger.info("%s", line*len(title))


def log_subtitle(subtitle):
    """Add visual breaks in the logging for sub sections."""
    log_title(subtitle, line='-', section=False)


def check_parameter_names(hdf):
    """
    Check if the parameter name is one recognised as a POLARIS parameter
    name.
    Returns two tuples.
      The first a tuple of names matching POLARIS parameters.
      The second, a tuple of names that do not match POLARIS parameters and
      will be ignored by analysis.
    """
    log_subtitle("Checking parameter names")
    hdf_parameters = set(hdf.keys())

    matched_names = set()
    for name in PARAMETER_LIST:
        if WILDCARD in name:
            found = wildcard_match(name, hdf_parameters, remove=' ')
        else:
            found = [name for p in hdf_parameters if p == name]
        if found:
            matched_names.update(found)

    unmatched_names = hdf_parameters - matched_names
    if not matched_names:
        logger.error("None of the %d parameters within HDF file are "
                     "recognised by POLARIS",
                     len(unmatched_names))
    elif unmatched_names:
        logger.info("Number of parameter names recognised by POLARIS: %s",
                    len(matched_names))
        logger.warn("Number of parameters names not recognised by "
                    "POLARIS: %s", len(unmatched_names))
        logger.debug("The following parameters names are recognised by "
                     "POLARIS: %s", matched_names)
        logger.debug("The following parameters names are not recognised by "
                     "POLARIS: %s", unmatched_names)
    else:
        logger.info("All %s parameters names are recognised by POLARIS.",
                    len(matched_names))
    return (tuple(matched_names), tuple(unmatched_names))


def check_for_core_parameters(hdf, helicopter=False):
    """
    Check that the following parameters exist in the file:
    - 'Airspeed'
    - 'Altitude STD'
    - either 'Heading' or 'Heading True'
    For Helicopters, an additional parameter, Rotor Speed, is required:
    - either 'Nr' or for dual rotors 'Nr (1)' and 'Nr (2)'
    Minimum parameter required for any analysis to be performed.
    """
    hdf_parameters = hdf.keys()
    airspeed = 'Airspeed' in hdf_parameters
    altitude = 'Altitude STD' in hdf_parameters
    heading = 'Heading' in hdf_parameters
    heading_true = 'Heading True' in hdf_parameters
    rotor = 'Nr' in hdf_parameters
    nr1and2 = ('Nr (1)' in hdf_parameters) and ('Nr (2)' in hdf_parameters)

    core_available = airspeed and altitude and (heading or heading_true)
    if core_available and helicopter:
        core_available = rotor or nr1and2

    if core_available:
        logger.info("All core parameters available for analysis.")
        if heading_true and not heading:
            logger.info("Analysis will use parameter 'Heading True' as "
                        "parameter 'Heading' not available.")
    else:
        if not airspeed:
            logger.error("Parameter 'Airspeed' not found. Required as one "
                         "of the core parameters for analysis.")
        if not altitude:
            logger.error("Parameter 'Altitude STD' not found. Required as "
                         "one of the core parameters for analysis.")
        if not heading and not heading_true:
            logger.error("Parameter 'Heading' and 'Heading True' not found. "
                         "One of these parameters is required for analysis.")
        if helicopter and not rotor and not nr1and2:
            logger.error("Parameter 'Nr' (or 'Nr (1)' and 'Nr (2)') not "
                         "found. Helicopter's rotor speed is required as one "
                         "of the core parameter for analysis.")
    return core_available


# =============================================================================
#   Parameter's Attributes
# =============================================================================
def validate_parameters(hdf, helicopter=False):
    """
    Iterates through all the parameters within the 'series' namespace and
    validates:
        Matches a POLARIS recognised parameter
        Attributes
        Data
    """
    log_title("Checking Parameters")
    matched, _ = check_parameter_names(hdf)
    check_for_core_parameters(hdf, helicopter)
    for name, parameter in hdf.iteritems():
        log_title("Checking Parameter: '%s'" % (name, ))
        if name in matched:
            logger.info("'%s' is a recognised by POLARIS.", name)
        else:
            logger.warn("'%s' is not a recognised by POLARIS.", name)
        if name in PARAMETERS_CORE:
            logger.info("'%s' is a core parameter required for basic "
                        "analysis.", name)
        validate_parameter_attributes(hdf, name, parameter, name in matched)
        validate_parameters_dataset(hdf, name, parameter)
    return


def validate_parameter_attributes(hdf, name, parameter, matched):
    """Validates all parameter attributes."""
    log_subtitle("Checking Attribute for Parameter: %s" % (name, ))
    validate_arinc_429(parameter)
    validate_data_type(parameter)
    validate_frequency(hdf, parameter)
    validate_lfl(parameter)
    validate_name(parameter, name)
    validate_source_name(parameter, matched)
    validate_supf_offset(parameter)
    validate_units(parameter)
    validate_values_mapping(hdf, parameter)


def validate_parameters_dataset(hdf, name, parameter):
    """Validates all parameter datasets."""
    log_subtitle("Checking dataset for Parameter: %s" % (name, ))
    validate_dataset(hdf, name, parameter)


# =============================================================================
#   Parameter's Attributes
# =============================================================================
def validate_arinc_429(parameter):
    """
    Reports if parameter attribute arinc_429 exists.
    If so, check it is a boolean and report it's value.
    """
    logger.info("Checking parameter attribute: arinc_429")
    if parameter.arinc_429 is None:
        logger.warn("No attribute 'arinc_429' for '%s'. Optional attribute, "
                    "if parmater does not have an ARINC 429 source.",
                    parameter.name)
    else:
        if 'bool' not in type(parameter.arinc_429).__name__:
            logger.error("Attribute 'arinc_429' is not a Boolean type.")
        if parameter.arinc_429:
            logger.info("'%s' has an ARINC 429 source.", parameter.name)
        else:
            logger.info("'%s' does not have an ARINC 429 source.",
                        parameter.name)


def validate_data_type(parameter):
    """
    Checks the parameter attribute data_type exists (It is required)
    and verify that the data has the correct type.
    """
    logger.info("Checking parameter attribute: data_type")
    if parameter.data_type is None:
        logger.error("No attribute 'data_type' present for '%s'. "
                     "This is required attribute.", parameter.name)
    else:
        logger.info("'%s' has a 'data_type' attribute of: %s", parameter.name,
                    parameter.data_type)
        logger.info("'%s' data has a dtype of: %s", parameter.name,
                    parameter.array.data.dtype)
        if parameter.data_type in ['ASCII', ]:
            if 'string' not in parameter.array.dtype.name:
                logger.error("'%s' data type is %s. It should be a string "
                             "for '%s' parameters.", parameter.name,
                             parameter.array.dtype.name, parameter.data_type)
                return
        elif parameter.data_type in ['BCD', 'Interpolated', 'Polynomial',
                                     'Signed', 'Synchro', 'Unsigned']:
            if 'float' not in parameter.array.dtype.name:
                logger.error("'%s' data type is %s. It should be a float "
                             "for '%s' parameters.", parameter.name,
                             parameter.array.dtype.name, parameter.data_type)
                return
        elif parameter.data_type in ['Multi-state', 'Discrete']:
            if 'int' not in parameter.array.dtype.name:
                logger.warn("'%s' data type is %s. It should be an integer "
                            "for '%s' parameters.", parameter.name,
                            parameter.array.dtype.name, parameter.data_type)
                return
        logger.info("'%s' data_type is %s and is an array of %s.",
                    parameter.name, parameter.data_type,
                    parameter.array.dtype.name)


def validate_frequency(hdf, parameter):
    """
    Checks the parameter attribute frequency exists (It is required)
    and report if it is a valid frequency and if it is listed in the root
    attribute frequencies.
    """
    logger.info("Checking parameter attribute: frequency")
    if parameter.frequency is None:
        logger.error("No attribute 'frequency' present for '%s'. "
                     "This is required attribute.", parameter.name)
    else:
        if parameter.frequency not in VALID_FREQUENCIES:
            logger.error("'%s' has a 'frequency' of %s which is not a "
                         "frequency supported by POLARIS.", parameter.name,
                         parameter.frequency)
        else:
            logger.info("'frequency' is %s Hz for '%s' and is a support "
                        "frequency.", parameter.frequency, parameter.name)
        if hdf.frequencies is not None:
            if 'array' in type(hdf.frequencies).__name__:
                if parameter.frequency not in hdf.frequencies:
                    logger.warn("Frequency not in the Root attribute list "
                                "of frequenices.")
            elif parameter.frequency != hdf.frequencies:
                logger.warn("Frequency not in the Root attribute list of "
                            "frequenices.")


def validate_lfl(parameter):
    '''
    Check that the required lfl attribute is present. Report if recorded or
    derived.
    '''
    logger.info("Checking parameter attribute: lfl")
    if parameter.lfl is None:
        logger.error("No attribute 'lfl' for '%s'. Attribute is Required.",
                     parameter.name)
        return
    if 'bool' not in type(parameter.lfl).__name__:
        logger.error("lfl should be an Boolean. Type is %s",
                     type(parameter.lfl).__name__)
    if parameter.lfl:
        logger.info("'%s' is a recorded parameter.", parameter.name)
    else:
        logger.info("'%s' is a derived parameter.", parameter.name)


def validate_name(parameter, name):
    """
    Checks the parameter attribute name exists (It is required)
    and report if name matches the parameter's group name.
    """
    logger.info("Checking parameter attribute: name")
    if parameter.name is None:
        logger.error("No attribute 'name' for '%s'. Attribute is Required.",
                     name)
    else:
        if parameter.name != name:
            logger.error("'name' is present, but is not the same name as "
                         "the parameter group. name: %s, parameter group: %s",
                         parameter.name, name)
        else:
            logger.info("'name' is present and name is the same name as "
                        "the parameter group.")


def validate_source_name(parameter, matched):
    """Reports if the parameter attribute source_name exists."""
    logger.info("Checking parameter attribute: source_name")
    if parameter.source_name is None:
        logger.info("No attribute 'source_name' for '%s'. Attribute is "
                    "optional.", parameter.name)
    else:
        try:
            pname = parameter.source_name.decode('utf8')
        except UnicodeDecodeError:
            pname = repr(parameter.source_name)
        add_msg = ''
        if matched:
            add_msg = 'POLARIS name '
        logger.info("'source_name' is present. Original name '%s' maps to "
                    "%s'%s'", pname, add_msg, parameter.name)


def validate_supf_offset(parameter):
    """
    Check if the parameter attribute supf_offset exists (It is required)
    and report.
    """
    logger.info("Checking parameter attribute: supf_offset")
    if parameter.offset is None:
        logger.error("No attribute 'supf_offset' for '%s'. Attribute is "
                     "Required. ", parameter.name)
    else:
        if 'float' not in type(parameter.offset).__name__:
            msg = "'supf_offset' type for '%s' is not a float. Got %s instead"\
                % (parameter.name, type(parameter.offset).__name__)
            if parameter.offset == 0:
                logger.warn(msg)
            else:
                logger.error(msg)
        else:
            logger.info("'supf_offset' is present and correct data type "
                        "and has a value of %s", parameter.offset)


def validate_units(parameter):
    """
    Check if the parameter attribute units exists (It is required)
    and reports the value and if it is valid unit name.
    """
    logger.info("Checking parameter attribute: units")
    if parameter.data_type in ('Discrete', 'Multi-state',
                               'Enumerated Discrete'):
        return
    if parameter.units is None:
        logger.warn("No attribute 'units' for '%s'. Attribute is Required.",
                    parameter.name)
    else:
        if type(parameter.units).__name__ not in ['str', 'string', 'string_']:
            logger.error("'units' expected to be a string, got %s",
                         type(parameter.units).__name__)
        if parameter.units == '':
            logger.info("Attribute 'units' is present for '%s', but empty.",
                        parameter.name)
        elif parameter.units in ut.available():
            logger.info("Attribute 'units' is present for '%s' and has a "
                        "valid unit of '%s'.", parameter.name, parameter.units)
        else:
            logger.error("Attribute 'units' is present for '%s', but has an "
                         "unknown unit of '%s'.",
                         parameter.name, parameter.units)


def validate_values_mapping(hdf, parameter):
    """
    Check if the parameter attribute values_mapping exists (It is required for
    discrete or multi-state parameter) and reports the value.
    """
    logger.info("Checking parameter attribute: values_mapping")
    if parameter.values_mapping is None:
        if parameter.data_type in ('Discrete', 'Multi-state',
                                   'Enumerated Discrete'):
            logger.error("No attribute 'values_mapping' for '%s'. "
                         "Attribute is Required for a %s parameter.",
                         parameter.name, parameter.data_type)
        else:
            logger.info("No attribute 'values_mapping' not required for '%s'.",
                        parameter.name)
    else:
        logger.info("Attribute, 'values_mapping' value is: %s",
                    parameter.values_mapping)
        try:
            # validate JSON string
            jstr = json.loads(
                hdf.hdf['/series/' + parameter.name].attrs['values_mapping']
            )
            logger.info("Attribute, 'values_mapping' is a valid json "
                        "string: %s", jstr)
        except ValueError as err:
            logger.error("'values_mapping' is not a valid JSON string. (%s)",
                         err)
        if parameter.data_type == 'Discrete':
            try:
                value0 = parameter.values_mapping[0]  # False values
                logger.debug("discrete value of 0 maps to '%s'.", value0)
            except KeyError:
                logger.debug("discrete value of 0 has no mapping.")
            try:
                value1 = parameter.values_mapping[1]  # True values
                if value1 in ["", "-"]:
                    logger.error("discrete value of 1 should map to a non "
                                 "emtpy (or '-') string value of 1. Got '%s'.",
                                 value1)
                else:
                    logger.debug("discrete value of 1 maps to '%s'", value1)
            except KeyError:
                logger.error("discrete value of 1 has no mapping. Needs "
                             "to have a mapping for this value.")
            if len(parameter.values_mapping.keys()) > 2:
                logger.error("'%s' a discrete parameter, but values_mapping "
                             "attribute has %s values should be no "
                             "more than 2.", parameter.name,
                             len(parameter.data_type.keys()))


def validate_dataset(hdf, name, parameter):
    """Check the data for size, unmasked inf/NaN values."""
    inf_nan_check(parameter)

    boundary = (64.0 if hdf.superframe_present else 4.0)
    # Expected size of the data is duration * the parameter's frequency,
    # includes any padding required to the next frame/super frame boundary
    expected_data_size = \
        ceil(hdf.duration / boundary) * boundary * parameter.frequency

    logger.info("Checking parameter actual dataset size against expected "
                "size (frame aligned, duration * parameter's frequency).")

    if expected_data_size != parameter.array.size:
        logger.error("The data size of '%s' is %s and different to expected "
                     "size of %s.",
                     parameter.name, parameter.array.size, expected_data_size)
    else:
        logger.info("Data size of '%s' is of the expected size of %s.",
                    parameter.name, expected_data_size)
    if parameter.array.data.size != parameter.array.mask.size:
        logger.error("The data and mask sizes are different. (Data is %s, "
                     "Mask is %s)", parameter.array.data.size,
                     parameter.array.mask.size)
    else:
        logger.info("Data and Mask both have the size of %s elements.",
                    parameter.array.data.size)

    logger.info("Checking dataset type and shape.")
    masked_array = isinstance(parameter.array, np.ma.core.MaskedArray)
    mapped_array = isinstance(parameter.array, MappedArray)
    if not masked_array and not mapped_array:
        logger.error("Data for %s is not a MaskedArray or MappedArray. "
                     "Type is %s", name, type(parameter.array))
    else:
        # check shape, it should be 1 dimensional arrays for data and mask
        if len(parameter.array.shape) != 1:
            logger.error("The data and mask are not in an 1 dimensional "
                         "array. The data's shape is %s ",
                         parameter.array.shape)
        else:
            logger.info("Data is in a %s with a shape of %s",
                        type(parameter.array).__name__, parameter.array.shape)


def inf_nan_check(parameter):
    '''
    Check the dataset for NaN or inf values
    '''
    def _report(count, parameter, unmasked, val_str):
        '''
        log as warning if all are masked, error if not
        '''
        if count:
            msg = "%s %s values found in the data of '%s'. " \
                % (count, val_str, parameter.name)
            nan_percent = (float(count)/len(parameter.array.data))*100
            msg += "Represents %.2f%%. " % (nan_percent, )
            if unmasked:
                msg += "%s are not masked." % (unmasked,)
                logger.error(msg)
            else:
                msg += "All are masked."
                logger.warn(msg)

    logger.info("Checking parameter dataset for inf and NaN values.")
    if 'int' in parameter.array.dtype.name or \
       'float' in parameter.array.dtype.name:

        nan_unmasked = np.ma.masked_equal(
            np.isnan(parameter.array), False).count()
        nan_count = np.ma.masked_equal(
            np.isnan(parameter.array.data), False).count()
        inf_unmasked = np.ma.masked_equal(
            np.isinf(parameter.array), False).count()
        inf_count = np.ma.masked_equal(
            np.isinf(parameter.array.data), False).count()

        _report(nan_count, parameter, nan_unmasked, 'NaN')
        _report(inf_count, parameter, inf_unmasked, 'inf')

        if nan_count == inf_count == 0:
            logger.info("dataset does not have any inf or NaN values.")


def validate_namespace(hdf5):
    '''Uses h5py functions to verify what is stored on the root group.'''
    found = ''
    log_title("Checking for the namespace 'series' group on root")
    if 'series' in hdf5.keys():
        logger.info("Found the POLARIS namespace 'series' on root.")
        found = 'series'
    else:
        found = [g for g in hdf5.keys() if 'series' in g.lower()]
        if found:
            # series found but in the wrong case.
            logger.error("Namespace '%s' found, but needs to be in "
                         "lower case.", found)
        else:
            logger.error("Namespace 'series' was not found on root.")

    logger.info("Checking for other namespace groups on root.")
    group_num = len(hdf5.keys())

    show_groups = False
    if group_num is 1 and 'series' in found:
        logger.info("Namespace 'series' is the only group on root.")
    elif group_num is 1 and 'series' not in found:
        logger.error("Only one namespace on root, but it's not 'series'.")
        show_groups = True
    elif group_num is 0:
        logger.error("No namespace groups in found in the file.")
    elif group_num > 1 and 'series' in found:
        logger.warn("Namespace 'series' found along with %s addtional "
                    "groups.", group_num-1)
        show_groups = True
    elif group_num > 1:
        logger.error("%s namespace groups are on root, but not 'series'.",
                     group_num)
        show_groups = True
    if show_groups:
        logger.debug("The following namespace groups are on root: %s",
                     [g for g in hdf5.keys() if 'series' not in g])
        logger.info("If these are parmeters they must be stored "
                    "within 'series'.")


# =============================================================================
#   Root Attributes
# =============================================================================
def validate_root_attribute(hdf):
    """Validates all the root attributes."""
    log_title("Checking the Root attributes")
    validate_duration_attribute(hdf)
    validate_frequencies_attribute(hdf)
    validate_reliable_frame_counter_attribute(hdf)
    validate_reliable_subframe_counter_attribute(hdf)
    validate_start_timestamp_attribute(hdf)
    validate_superframe_present_attribute(hdf)


def validate_duration_attribute(hdf):
    """
    Check if the root attribute duration exists (It is required)
    and report the value.
    """
    logger.info("Checking Root Attribute: duration")
    if hdf.duration:
        logger.info("duration attribute present with a value of %s.",
                    hdf.hdf.attrs['duration'])
        if 'int' in type(hdf.hdf.attrs['duration']).__name__:
            logger.debug("duration attribute is an integer.")
        else:
            logger.error("duration attribute is not an integer, type "
                         "reported as '%s'.",
                         type(hdf.hdf.attrs['duration']).__name__)
    else:
        logger.error("No root attribrute 'duration'. This is a required "
                     "attribute.")


def validate_frequencies_attribute(hdf):
    """
        Check if the root frequencies duration exists.
        report all the values and if the list covers all frequencies used
        by the store parameters.
    """
    logger.info("Checking Root Attribute: frequencies")
    if hdf.frequencies is not None:
        logger.info("frequencies attribute present.")
        name = type(hdf.frequencies).__name__
        if 'array' in name or 'list' in name:
            floatcount = 0
            rootfreq = set(list(hdf.frequencies))
            for value in hdf.frequencies:
                if 'float' in type(value).__name__:
                    floatcount += 1
                else:
                    logger.error("frequency value %s should be a float",
                                 value)
            if floatcount == len(hdf.frequencies):
                logger.info("Passed: frequencies listed are float values.")
            else:
                logger.error("Some or all frequencies are not float values.")
        elif 'float' in name:
            logger.info("Passed: frequency listed is a float value.")
            rootfreq = set([hdf.frequencies])
        else:
            logger.error("frequency listed is not a float value.")

        paramsfreq = set([v.frequency for _, v in hdf.iteritems()])
        if rootfreq == paramsfreq:
            logger.info("Root frequency list covers all the frequencies "
                        "used by parameters.")
        elif rootfreq - paramsfreq:
            logger.info("Root frequencies lists has more frequencies than "
                        "used by parameters. Unused frquencies: %s",
                        list(rootfreq - paramsfreq))
        elif paramsfreq - rootfreq:
            logger.info("More parameter frequencies used than listed in root "
                        "attribute frequencies. Frequency not listed: %s",
                        list(paramsfreq - rootfreq))
    else:
        logger.info("frequencies attribute not present and is optional.")


def is_reliable_frame_counter(hdf):
    """returns if the parameter 'Frame Counter' is reliable."""
    try:
        pfc = hdf['Frame Counter']
    except KeyError:
        return False
    if np.ma.masked_inside(pfc, 0, 4095).count() != 0:
        return False
    # from split_hdf_to_segments.py
    fc_diff = np.ma.diff(pfc.array)
    fc_diff = np.ma.masked_equal(fc_diff, 1)
    fc_diff = np.ma.masked_equal(fc_diff, -4095)
    if fc_diff.count() == 0:
        return True
    return False


def validate_reliable_frame_counter_attribute(hdf):
    """
    Check if the root attribute reliable_frame_counter exists (It is required)
    and report the value and if the value is correctly set.
    """
    logger.info("Checking Root Attribute: 'reliable_frame_counter'")
    parameter_exists = 'Frame Counter' in hdf.keys()
    reliable = is_reliable_frame_counter(hdf)
    attribute_value = hdf.reliable_frame_counter
    correct_type = isinstance(hdf.reliable_frame_counter, bool)
    if attribute_value is None:
        logger.error("'reliable_frame_counter' attribute not present "
                     "and is required.")
    else:
        logger.info("'reliable_frame_counter' attribute is present.")
        if parameter_exists and reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and is "
                        "reliable and 'reliable_frame_counter' marked "
                        "as reliable.")
        elif parameter_exists and reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and is "
                        "reliable, but 'reliable_frame_counter' not "
                        "marked as reliable.")
        elif parameter_exists and not reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and not "
                        "reliable, but 'reliable_frame_counter' is marked "
                        "as reliable.")
        elif parameter_exists and not reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and not "
                        "reliable and 'reliable_frame_counter' not marked "
                        "as reliable.")
        elif not parameter_exists and attribute_value:
            logger.info("'Frame Counter' parameter not present, but "
                        "'reliable_frame_counter' marked as reliable. "
                        "Value should be False.")
        elif not parameter_exists and not attribute_value:
            logger.info("'Frame Counter' parameter not present and "
                        "'reliable_frame_counter' correctly set to False.")
        if not correct_type:
            logger.error("'reliable_frame_counter' is not a Boolean type. "
                         "Type is %s",
                         type(hdf.reliable_frame_counter).__name__)


def is_reliable_subframe_counter(hdf):
    """returns if the parameter 'Subframe Counter' is reliable."""
    try:
        sfc = hdf['Subframe Counter']
    except KeyError:
        return False
    sfc_diff = np.ma.masked_equal(np.ma.diff(sfc.array), 1)
    if sfc_diff.count() == 0:
        return True
    return False


def validate_reliable_subframe_counter_attribute(hdf):
    """
    Check if the root attribute reliable_subframe_counter exists
    (It is required) and report the value and if the value is correctly set.
    """
    logger.info("Checking Root Attribute: 'reliable_subframe_counter'")
    parameter_exists = 'Subframe Counter' in hdf.keys()
    reliable = is_reliable_subframe_counter(hdf)
    attribute_value = hdf.reliable_subframe_counter
    correct_type = isinstance(hdf.reliable_subframe_counter, bool)
    if attribute_value is None:
        logger.error("'reliable_subframe_counter' attribute not present "
                     "and is required.")
    else:
        logger.info("'reliable_subframe_counter' attribute is present.")
        if parameter_exists and reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and is "
                        "reliable and 'reliable_subframe_counter' marked "
                        "as reliable.")
        elif parameter_exists and reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and is "
                        "reliable, but 'reliable_subframe_counter' not "
                        "marked as reliable.")
        elif parameter_exists and not reliable and attribute_value:
            logger.info("'Frame Counter' parameter is present and not "
                        "reliable, but 'reliable_subframe_counter' is "
                        "marked as reliable.")
        elif parameter_exists and not reliable and not attribute_value:
            logger.info("'Frame Counter' parameter is present and not "
                        "reliable and 'reliable_subframe_counter' not "
                        "marked as reliable.")
        elif not parameter_exists and attribute_value:
            logger.info("'Frame Counter' parameter not present, but "
                        "'reliable_subframe_counter' marked as reliable. "
                        "Value should be False.")
        elif not parameter_exists and not attribute_value:
            logger.info("'Frame Counter' parameter not present and "
                        "'reliable_subframe_counter' correctly set to False.")
        if not correct_type:
            logger.error("'reliable_subframe_counter' is not a Boolean type. "
                         "Type is %s",
                         type(hdf.reliable_subframe_counter).__name__)


def validate_start_timestamp_attribute(hdf):
    """
    Check if the root attribute start_timestamp exists
    and report the value.
    """
    logger.info("Checking Root Attribute: start_timestamp")
    if hdf.start_datetime:
        logger.info("start_timestamp attribute present.")
        logger.info("Time reported to be, %s", hdf.start_datetime)
        logger.info("Epoch timestamp value is: %s",
                    hdf.hdf.attrs['start_timestamp'])
        if 'float' in type(hdf.hdf.attrs['start_timestamp']).__name__:
            logger.info("start_timestamp is a float.")
        else:
            logger.error("'start_timestamp' attribute is not a float, type "
                         "reported as '%s'.",
                         type(hdf.hdf.attrs['start_timestamp']).__name__)
    else:
        logger.info("'start_timestamp' attribute not present and is optional.")


def validate_superframe_present_attribute(hdf):
    """
    Check if the root attribute superframe_present exists
    and report.
    """
    logger.info("Checking Root Attribute: superframe_present.")
    if hdf.superframe_present:
        logger.info("superframe_present attribute present.")
    else:
        logger.info("superframe_present attribute is not present and "
                    "is optional.")


def validate_file(hdffile, helicopter=False):
    """
    Attempts to open the HDF5 file in using FlightDataAccessor and run all the
    validation tests. If the file cannot be opened, it will attempt to open
    the file using the h5py package and validate the namespace to test the
    HDF5 group structure.
    """
    filename = hdffile.split(os.sep)[-1]
    open_with_h5py = False
    hdf = None
    logger.info("Verifying file '%s' with FlightDataAccessor.", filename)
    try:
        hdf = hdf_file(hdffile, read_only=True)
    except Exception as err:
        logger.error("FlightDataAccessor cannot open '%s'. "
                     "Exception(%s: %s)", filename, type(err).__name__, err)
        open_with_h5py = True
    # If FlightDataAccessor errors upon opening it maybe because '/series'
    # is not included in the file. hdf_file attempts to create and fails
    # because we opening it as readonly. Verify the group structure by
    # using H5PY
    if open_with_h5py:
        logger.info("Checking that H5PY package can read the file.")
        try:
            hdf_alt = h5py.File(hdffile, 'r')
        except Exception as err:
            logger.error("cannot open '%s' using H5PY. Exception(%s: %s)",
                         filename, type(err).__name__, err)
            return
        logger.info("File %s can be opened by H5PY, suggesting the format "
                    "is not compatible for POLARIS to use.",
                    filename)
        logger.info("Will just verify the HDF5 structure and exit.")
        validate_namespace(hdf_alt)
        hdf_alt.close()
    else:
        validate_namespace(hdf.hdf)
        # continue testing using FlightDataAccessor
        validate_root_attribute(hdf)
        validate_parameters(hdf, helicopter)
    if hdf:
        hdf.close()


def main():
    """Main"""
    parser = argparse.ArgumentParser(
        description="Flight Data Services, HDF5 Validator for POLARIS "
                    "compatibility.",
        version="1.3"
    )
    parser.add_argument(
        '--helicopter',
        help='Validates HDF5 file against helicopter core parameters.',
        action='store_true',
    )
    parser.add_argument(
        "-s",
        "--stop",
        help="Stop validation on the first error encountered.",
        action="store_true"
    )
    parser.add_argument(
        "-e",
        "--error",
        help="Display only errors on screen.",
        action="store_true"
    )
    #parser.add_argument(
        #"-w",
        #"--warn",
        #help="Display only warnings and errors on screen.",
        #action="store_true"
    #)
    parser.add_argument(
        '-o',
        '--output',
        metavar='LOG',
        type=str,
        help='Saves all the screen messages, during validation to a log file.',
    )
    parser.add_argument(
        'HDF5',
        help="The HDF5 file to be tested for POLARIS compatibility. "
        "This will also compressed hdf5 files with the extension '.gz'.)"
    )
    args = parser.parse_args()

    # Setup logger
    fmtr = logging.Formatter(r'%(levelname)-9s:%(message)s')

    # setup a output file handler, if required
    if args.output:
        logfilename = args.output
        file_hdlr = logging.FileHandler(logfilename, mode='w')
        file_hdlr.setLevel(logging.DEBUG)
        file_hdlr.setFormatter(fmtr)
        logger.addHandler(file_hdlr)

    # setup a stream handler (display to terminal)
    term_hdlr = logging.StreamHandler()
    if args.error:
        term_hdlr.setLevel(logging.ERROR)
    else:
        term_hdlr.setLevel(logging.INFO)
    term_hdlr.setFormatter(fmtr)
    logger.addHandler(term_hdlr)

    # setup a separate handler so we count log levels and stop on first error
    hdfv_hdlr = HDFValidatorHandler(args.stop)
    error_count = hdfv_hdlr.get_error_counts()
    hdfv_hdlr.setLevel(logging.INFO)
    hdfv_hdlr.setFormatter(fmtr)
    logger.addHandler(hdfv_hdlr)

    logger.setLevel(logging.DEBUG)
    logger.debug("Arguments: %s", str(args))
    try:
        validate_file(args.HDF5, args.helicopter)
    except StoppedOnFirstError:
        msg = "First error encountered. Stopping as requested."
        logger.info(msg)
        if args.error:
            print msg

    for hdr in logger.handlers:
        if isinstance(hdr, HDFValidatorHandler):
            error_count = hdr.get_error_counts()

    log_title("Results")
    msg = "Validation ended with, %s errors and %s warnings" %\
        (error_count['errors'], error_count['warnings'])
    logger.info(msg)
    if args.error:
        print msg

if __name__ == '__main__':
    main()
