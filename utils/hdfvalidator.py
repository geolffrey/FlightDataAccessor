import h5py
import argparse
import os
import sys
import json
import numpy as np
from hdfaccess.file import hdf_file
from analysis_engine.utils import list_parameters
from flightdatautilities import units as ut

from collections import Counter

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

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

def title(title, line='-'):
    print "%s" % ('_'*80)
    print title
    print "%s" % (line*len(title),)

def check_parameter_names(hdf):
    params_from_file = set(hdf.keys())
    parameter_naming = set(list_parameters()) | set(EXTRA_PARAMETERS)
    matched_names = set()
    for name in parameter_naming:
        found = hdf.search(name)
        if found:
            matched_names.update(found)
    unmatched_names = params_from_file - matched_names
    if unmatched_names:
        print "Number of parameter recongnised by POLARIS: %s" \
              % (len(matched_names),)
        print "Number of parameters unrecongnised by POLARIS: %s" \
              % (len(unmatched_names),)
        print "The following parameters names are unrecognised by "\
              "POLARIS: %s" % (unmatched_names,)
    else:
        print "All %s parameters are recongnised by POLARIS." \
              % (len(matched_names),)
        
def check_for_core_parameters(hdf):
    params_from_file = hdf.keys()
    airspeed = 'Airspeed' in params_from_file
    altitude = 'Altitude STD' in params_from_file
    heading = 'Heading' in params_from_file
    heading_true = 'Heading True' in params_from_file
    
    if airspeed and altitude and (heading or heading_true):
        print "Passed: All core parameters available for analysis."
        if heading_true and not heading:
            print "Analysis will use 'Heading True' as no 'Heading' available."
        return 
    else:
        if not airspeed:
            print "Error: 'Airspeed' not found."
            result['failed'] += 1
        if not altitude:
            print "Error: 'Altitude STD' not found."
            result['failed'] += 1
        if not heading and not heading_true:
            print "Error: 'Heading' and 'Heading True' not found."
            result['failed'] += 1
            
#==============================================================================
#   Parameter's Attributes
#==============================================================================
def validate_parameter_attributes(hdf):
    for name, parameter in hdf.iteritems():
        title("Checking Attribute for Parameter: %s" % (name))
        validate_arinc_429(hdf, name, parameter)
        validate_data_type(hdf, name, parameter)
        validate_frequency(hdf, name, parameter)
        validate_lfl(hdf, name, parameter)
        validate_name(hdf, name, parameter)
        validate_source_name(hdf, name, parameter)
        validate_supf_offset(hdf, name, parameter)
        validate_units(hdf, name, parameter)
        validate_values_mapping(hdf, name, parameter)
        validate_dataset(hdf, name, parameter)

def validate_arinc_429(hdf, name, parameter):
    print "Checking parameter attribute: arinc_429"
    if parameter.arinc_429 is None:
        print "No attribute 'arinc_429'. '%s' does not have an ARINC "\
              "429 source." % name
        return
    else:
        if 'bool' not in type(parameter.arinc_429).__name__:
            print "Error: Attribute 'arinc_429' is not a Boolean type."
            result['failed'] += 1
        if parameter.arinc_429:
            print "'%s' has an ARINC 429 source." % name
        else:
            print "'%s' does not have an ARINC 429 source." % name

def validate_data_type(hdf, name, parameter):
    print "Checking parameter attribute: data_type"
    if parameter.data_type is None:
        print "Error: No attribute 'data_type' present for '%s'. "\
              "This is required attribute." % (name,)
        result['failed'] += 1
        return 
    else:
        print "'%s' has a 'data_type' attribute of: %s" \
              % (name, parameter.data_type)
        print "'%s' data has a dtype of: %s" \
              % (name, parameter.array.data.dtype)
        if parameter.data_type in ['ASCII',]:
            if 'string' not in parameter.array.dtype.name:
                print "Error: Should be a string for '%s' parameters." \
                      % (parameter.data_type,)
                result['failed'] += 1
                return
        elif parameter.data_type in ['BCD', 'Interpolated', 'Polynomial',
                                     'Signed', 'Synchro', 'Unsigned']:
            if 'float' not in parameter.array.dtype.name:
                print "Error: Should be a float for '%s' parameters." \
                      % (parameter.data_type,)
                result['failed'] += 1
                return
        elif parameter.data_type in ['Multi-state', 'Discrete']:
            if 'int' not in parameter.array.dtype.name:
                print "Error: Should be an integer for '%s' parameters." \
                      % (parameter.data_type,)
                result['failed'] += 1
                return 
        print "'%s' data_type is %s and is an array of %s." \
              % (name, parameter.data_type, parameter.array.dtype.name)         
   

def validate_frequency(hdf, name, parameter):
    print "Checking parameter attribute: frequency"
    if parameter.frequency is None:
        print "Error: No attribute 'frequency' present for '%s'. "\
              "This is required attribute." % (name,)
        result['failed'] += 1
        return
    if parameter.frequency not in VALID_FREQUENCIES:
        print "Error: '%s' has a 'frequency' of %s which is not a "\
              "frequency supported by POLARIS." % (name, parameter.frequency)
        result['failed'] += 1
    else:
        print "'frequency' is %s Hz for '%s' is a support frequency." \
              % (parameter.frequency, name)
    if hdf.frequencies is not None:
        if 'array' in type(hdf.frequencies).__name__:
            if parameter.frequency not in hdf.frequencies:
                print "Frequency not in the Root attribute list of frequenices."
        elif parameter.frequency != hdf.frequencies:
            print "Frequency not in the Root attribute list of frequenices."

def validate_lfl(hdf, name, parameter):
    ''' 
    Check that the required lfl attribute is present. Report if recored or 
    derived.
    '''
    print "Checking parameter attribute: lfl"
    if parameter.lfl is None:
        print "Error: No attribute 'lfl' for '%s'. Attribute is Required."\
              % (name,)
        result['failed'] += 1
        return
    if 'bool' not in type(parameter.lfl).__name__:
        print "Error: lfl should be an Boolean. Type is %s" \
              % (type(parameter.lfl).__name__,)
        result['failed'] += 1
    if parameter.lfl:
        print "'%s' is a recorded parameter." % (name,)
    else:
        print "'%s' is a derived parameter." % (name,)

def validate_name(hdf, name, parameter):
    print "Checking parameter attribute: name"
    if parameter.name is None:
        print "Error: No attribute 'name' for '%s'. Attribute is Required."\
              % (name,)
        result['failed'] += 1
        return
    else:
        if parameter.name != name:
            print "Error: 'name' is present, but is not the same name as "\
                  "the parameter group. name: %s, parameter group: %s" \
                  % (parameter.name, name)
            result['failed'] += 1
        else:
            print "'name' is present and name is the same name as "\
                  "the parameter group."


def validate_source_name(hdf, name, parameter):
    print "Checking parameter attribute: source_name"
    if parameter.source_name is None:
        print "No attribute 'source_name' for '%s'. Attribute is optional. "\
              % (name,)
        return
    else:
        print "'source_name' is present. Original name %s maps to "\
              "POLARIS name %s" % (parameter.source_name, name)

def validate_supf_offset(hdf, name, parameter):
    print "Checking parameter attribute: supf_offset"
    if parameter.offset is None:
        print "Error: No attribute 'supf_offset' for '%s'. "\
              "Attribute is Required. " % (name,)
        result['failed'] += 1
        return
    else:
        if 'float' not in type(parameter.offset).__name__:
            print "Error: 'supf_offset' type is not a float."
            result['failed'] += 1
        else:
            print "'supf_offset' is present and correct data type."
    

def validate_units(hdf, name, parameter):
    print "Checking parameter attribute: units"
    if 'string' not in type(parameter.units).__name__:
        print "Error: 'units' expected to be a string, got %s" \
              % (type(parameter.units).__name__)
    if parameter.units is None:
        print "Error: No attribute 'units' for '%s'. Attribute is Required."\
              % (name,)
        result['failed'] += 1
        return
    elif parameter.units is "":
        print "Attribute 'units' is present for '%s', but empty."\
              % (name,)
        return        
    elif parameter.units in ut.available():
        print "Attribute 'units' is present for '%s' and has a valid unit "\
              "of '%s'." % (name, parameter.units)
        return   
    elif parameter.units not in ut.available():
        print "Error: Attribute 'units' is present for '%s' and has an "\
              "unknown unit of '%s'." % (name, parameter.units)
        result['failed'] += 1
        return

def validate_values_mapping(hdf, name, parameter):
    print "Checking parameter attribute: values_mapping"
    if parameter.values_mapping is None:
        if parameter.data_type in ('Discrete', 'Multi-state',
                                   'Enumerated Discrete'):    
            print "Error: No attribute 'values_mapping' for '%s'. "\
                  "Attribute is Required for a $s parameter. "\
                  % (name, parameter.data_type)
            result['failed'] += 1
            return
        else:
            print "No attribute 'values_mapping' not required for '%s'. "\
                  % (name, )            
    else:
        print "Attribute, 'values_mapping' value is: %s"\
              % (parameter.values_mapping,)
        try:
            # validate JSON string
            jstr = json.loads(
                hdf.hdf['/series/' + name].attrs['values_mapping']
            )
            print "Attribute, 'values_mapping' is a valid json string: %s" % (jstr,)
        except ValueError as e:
            print "Error: 'values_mapping' is not a valid JSON string. (%s)"\
                  % (e, )
            result['failed'] += 1


def validate_dataset(hdf, name, parameter):
    print "Checking parameter dataset."
    if 'int' in parameter.array.dtype.name or \
       'float' in parameter.array.dtype.name:
        if np.ma.masked_equal(np.isnan(parameter.array),False).count() != 0:
            print "Error: NaN values found in data."
            result['failed'] += 1
        if np.ma.masked_equal(np.isinf(parameter.array),False).count() != 0:
            print "Error: Infinite values found in data."
            result['failed'] += 1
    expected_sized = hdf.duration * parameter.frequency
    actual_data_size = len(parameter.array)
    if expected_sized != actual_data_size:
        print "Error: The data size of '%s' different to expected size."
        result['failed'] += 1
    else:
        print "Data array is of the expected size."
    if len(parameter.array.data) != len(parameter.array.mask):
        print "Error: The data and mask sizes of '%s' are different."
        result['failed'] += 1
    else:
        print "Data and Mask both have the size of %s elements." \
              % (len(parameter.array.data))
    #print "%s" % (parameter.array.data.shape)
    #TODO: other tests

def validate_root_group(hdf5):
    '''Uses h5py functions to verify what is stored on the root group'''
    found = ''
    title("Checking for the namespace 'series' group on root", '=')
    if 'series' in hdf5.keys():
        print "Passed: 'series' found on root."
        found = 'series'
        result['passed'] += 1
    else:
        found = [g for g in hdf5.keys() if 'series' in g.lower()]
        if found:
            print "Error: '%s' was found, but needs to be in lower case." \
                  % (found,)
            result['failed'] += 1
        else:
            print "Error: 'series' was not found on root."
            result['failed'] += 1
                
    print "Checking group on root..."
    group_num = len(hdf5.keys())
    
    show_groups = False
    if group_num is 1 and 'series' in found:
        print "Passed: 'series' is the only group on root."
        result['passed'] += 1
    elif group_num is 1 and 'series' not in found:
        print "Error: Only one group on root, but it's not 'series'."
        result['failed'] += 1
        show_groups = True
    elif group_num is 0:
        print "Error: No groups in found in the file."
        result['failed'] += 1
    elif group_num > 1 and 'series' in found:
        print "Warning: 'series' found along with %s addtional groups. "\
              "There should only be 1 group named 'series'." % (group_num-1,)
        result['warning'] += 1
        show_groups = True
    elif group_num > 1:
        print "Error: %s groups were found on root. " % (group_num,)
        result['failed'] += 1
        show_groups = True
    if show_groups:
        print "The following groups are on root %s" \
              % ([g for g in hdf5.keys() if 'series' not in g],)
        print "If these are parmeters they must be stored within 'series'."
        
#==============================================================================
#   Root Attributes
#==============================================================================
def validate_root_attribute(hdf, stop_on_error):
    title("Checking the Root attributes", '=')
    validate_duration_attribute(hdf)
    validate_frequencies_attribute(hdf)
    validate_reliable_frame_counter_attribute(hdf)
    validate_reliable_subframe_counter_attribute(hdf)
    validate_start_timestamp_attribute(hdf)
    validate_superframe_present_attribute(hdf)


def validate_duration_attribute(hdf):
    print "Checking Root Attribute: duration"
    if hdf.duration:
        print "Passed: duration attribute present with a value of %s." \
              % (hdf.duration,)
        result['passed'] += 1
        if 'int' in type(hdf.hdf.attrs['duration']).__name__:
            print "Passed: duration is an integer."
            result['passed'] += 1
        else:
            print "Error: duration is not an integer, type reported as '%s'." \
                  % (type(hdf.hdf.attrs['duration']).__name__,)
            result['failed'] += 1        
    else:
        print "Error: No root attribrute 'duration'. "\
              "This is a required attribute"
        result['failed'] += 1

def validate_frequencies_attribute(hdf):
    print "Checking Root Attribute: frequencies"
    if hdf.frequencies is not None:
        print "frequencies attribute present."
        name = type(hdf.frequencies).__name__
        if 'array' in name or 'list' in name:
            floatcount = 0
            rf = set(list(hdf.frequencies))
            for value in hdf.frequencies:
                if 'float' in type(value).__name__:
                    floatcount += 1
                else:
                    print "Error: frequency value %s should be a float" \
                          % (value)
                    result['failed'] += 1
            if floatcount == len(hdf.frequencies):
                print "Passed: frequencies listed are float values."
            else:
                print "Error: some or all frequencies are not float values."
                result['failed'] += 1
        elif 'float' in name:
            print "Passed: frequency listed is a float value."
            rf = set([hdf.frequencies])
        else:
            print "Error: frequency listed is not a float value."
            result['failed'] += 1
            
        
        pf = set([v.frequency for k,v in hdf.iteritems()])
        if rf == pf:
            print "Root frequency list covers all the frequencies "\
                  "used by parameters."
        elif rf - pf:
            print "Root frequencies lists has more frequencies than used "\
                  "by parameters. Unused frquencies: %s" % (list(rf - pf),)
        elif pf - rf:
            print "More parameter frequencies used than listed in root "\
                  "attribute frequencies. Frequency not listed: %s" \
                  % (list(pf - rf),)        
    else:
        print "frequencies attribute not present and is optional."

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
    print "Checking Root Attribute: 'reliable_frame_counter'"
    parameter_exists = 'Frame Counter' in hdf.keys()
    reliable = validate_reliable_frame_counter(hdf)
    attribute_value = hdf.reliable_frame_counter
    correct_type = type(hdf.reliable_frame_counter) == bool
    if attribute_value is None:
        print "Error: 'reliable_frame_counter' attribute not present "\
              "and is required."
        result['failed'] += 1
    else:
        print "'reliable_frame_counter' attribute is present."
        if parameter_exists and reliable and attribute_value:
            print "'Frame Counter' parameter is present and is reliable "\
                  "and 'reliable_frame_counter' marked as reliable."
        elif parameter_exists and reliable and not attribute_value:
            print "'Frame Counter' parameter is present and is reliable, "\
                  "but 'reliable_frame_counter' not marked as reliable."      
        elif parameter_exists and not reliable and attribute_value:
            print "'Frame Counter' parameter is present and not reliable, "\
                  "but 'reliable_frame_counter' is marked as reliable."
        elif parameter_exists and not reliable and not attribute_value:
            print "'Frame Counter' parameter is present and not reliable "\
                  "and 'reliable_frame_counter' not marked as reliable."
        elif not parameter_exists and attribute_value:
            print "'Frame Counter' parameter not present, but "\
                  "'reliable_frame_counter' marked as reliable. "\
                  "Value should be False."
        elif not parameter_exists and not attribute_value:
            print "'Frame Counter' parameter not present and "\
                  "'reliable_frame_counter' correctly set to False."        

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
    print "Checking Root Attribute: 'reliable_subframe_counter'"
    parameter_exists = 'Subframe Counter' in hdf.keys()
    reliable = validate_reliable_subframe_counter(hdf)
    attribute_value = hdf.reliable_subframe_counter
    correct_type = type(hdf.reliable_subframe_counter) == bool
    if attribute_value is None:
        print "Error: 'reliable_subframe_counter' attribute not present "\
              "and is required."
        result['failed'] += 1
    else:
        print "'reliable_subframe_counter' attribute is present."
        if parameter_exists and reliable and attribute_value:
            print "'Frame Counter' parameter is present and is reliable "\
                  "and 'reliable_subframe_counter' marked as reliable."
        elif parameter_exists and reliable and not attribute_value:
            print "'Frame Counter' parameter is present and is reliable, "\
                  "but 'reliable_subframe_counter' not marked as reliable."      
        elif parameter_exists and not reliable and attribute_value:
            print "'Frame Counter' parameter is present and not reliable, "\
                  "but 'reliable_subframe_counter' is marked as reliable."
        elif parameter_exists and not reliable and not attribute_value:
            print "'Frame Counter' parameter is present and not reliable "\
                  "and 'reliable_subframe_counter' not marked as reliable."
        elif not parameter_exists and attribute_value:
            print "'Frame Counter' parameter not present, but "\
                  "'reliable_subframe_counter' marked as reliable. "\
                  "Value should be False."
        elif not parameter_exists and not attribute_value:
            print "'Frame Counter' parameter not present and "\
                  "'reliable_subframe_counter' correctly set to False."  

def validate_start_timestamp_attribute(hdf):
    print "Checking Root Attribute: start_timestamp"
    if hdf.start_datetime:
        print "start_timestamp attribute present."
        print "Time reported to be, %s" % (hdf.start_datetime)
        print "Epoch timestamp value is: %s" \
              % (hdf.hdf.attrs['start_timestamp'],)
        if 'float' in type(hdf.hdf.attrs['start_timestamp']).__name__:
            print "Passed: start_timestamp is a float."
        else:
            print "Error: start_timestamp is not a float, type reported "\
                  "as '%s'." \
                  % (type(hdf.hdf.attrs['start_timestamp']).__name__,)
            result['failed'] += 1
    else:
        print "'start_timestamp' attribute not present and is optional."

def validate_superframe_present_attribute(hdf):
    print "Checking Root Attribute: superframe_present."
    if hdf.superframe_present:
        print "superframe_present attribute present."
    else:
        print "superframe_present attribute is not present and is optional."


def validate_file(hdffile, stop_on_error):
    filename = hdffile.split(os.sep)[-1]
    open_with_h5py = False
    print "Verifying '%s' with FlightDataAccessor." % (filename,)
    try:
        hdf = hdf_file(hdffile, read_only=True)
        result['passed'] += 1
    except Exception as e:
        print "%s: %s" % (type(e).__name__, e)
        print "Error: FlightDataAccessor cannot open '%s'." % (filename,)
        result['failed'] += 1
        print "Checking file with H5PY package."
        open_with_h5py = True
    # If FlightDataAccessor errors upon opening it maybe because '/series'
    # is not included in the file. hdf_file attempts to create and fails 
    # because we opening it as readonly. Verify the group structure by 
    # using H5PY
    if open_with_h5py:
        try:
            hdf = h5py.File(hdffile, 'r')
            print "%s can be opened by H5PY package, suggesting the HDF5 "\
                  "file is not compatible for POLARIS" % (filename,)
        except Exception as e:
            print "%s: %s" % (type(e).__name__, e)
            print "H5PY cannot open '%s'." % (filename,)
            return
        validate_root_group(hdf)
    else:
        validate_root_group(hdf.hdf)
        #continue testing using FlightDataAccessor
        validate_root_attribute(hdf, stop_on_error)
        check_parameter_names(hdf)
        check_for_core_parameters(hdf)
        validate_parameter_attributes(hdf)
    title("Results", '=')
    print "Validation ended with %s passes, %s errors and %s warnings" \
          % (result['passed'], result['failed'], result['warning'])
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
        help="Will stop on the first issue encountered.",
        action="store_true"
    )
    parser.add_argument(
        "-r",
        "--report",
        help="Generate a report file.",
        metavar='FILE'
    )
    parser.add_argument(
        'file',
        help="Input HDF5 to be tested for POLARIS compatibility."
    )    
    args = parser.parse_args()
    if args.report:
        Tee(args.report, 'w')
    validate_file(args.file, args.stop)



if __name__ == '__main__':
    main()