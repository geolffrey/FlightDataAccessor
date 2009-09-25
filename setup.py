#!/usr/bin/env python

# The purpose of Skeleton is to serve as an example of a basic Python package
# including SetupTools capabilities. Comments have been adapted from
#
#  - http://pypi.python.org/pypi/setuptools
#  - http://peak.telecommunity.com/DevCenter/setuptools
#  - http://docs.python.org/distutils/
#  - http://docs.cython.org/docs/tutorial.html
#  - http://ianbicking.org/docs/setuptools-presentation/
#  - http://tarekziade.wordpress.com/2009/09/12/static-metadata-for-distutils/
#  - http://github.com/ella/setuptools-dummy
#
# TODO
#  - Cython extension modules
#  - Platform specific requirements
#  - pip compatibility
#  - Distribute compatibility

import sys
 
try:
    from setuptools import setup, find_packages
except ImportError:
    try:
        from ez_setup import use_setuptools
    except ImportError:
        print "can't find ez_setup"
        print "try: wget http://peak.telecommunity.com/dist/ez_setup.py"
        sys.exit(1)
    use_setuptools()
    from setuptools import setup, find_packages

setup(
    # Required meta data
    name='Skeleton',
    version='0.1.0',
    url='http://www.flightdataservices.com/',
    
    # Optional meta data   
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',            
    description='A Skeleton Python Package with DistUtils script',    
    long_description='''
    A Skeleton Python Package with DistUtils script you can
    use reStructuredText here
    ''',    
    download_url='http://www.flightdataservices.com/download/',
    classifiers='',
    platforms='',
    license='',

    # The packages option tells the Distutils to process (build, distribute, 
    # install, etc.) all pure Python modules found in each package mentioned in 
    # the packages list. In order to do this, of course, there has to be a 
    # correspondence between package names and directories in the filesystem. 
    # The default correspondence is the most obvious one, i.e. package distutils 
    # is found in the directory distutils relative to the distribution root. 
    
    # Thus, when you say packages = ['skeleton'] in your setup script, you are 
    # promising that the Distutils will find a file skeleton/__init__.py (which 
    # might be spelled differently on your system, but you get the idea) 
    # relative to the directory where your setup script lives. If you break this 
    # promise, the Distutils will issue a warning but still process the broken 
    # package anyways.
    packages=['skeleton'],
      
    # For a small module distribution, you might prefer to list all modules 
    # rather than listing packages-especially the case of a single module that 
    # goes in the "root package" (i.e., no package at all). 

    # This describes two modules, one of them in the "root" package, the other 
    # in the ivory package. Again, the default package/directory layout implies 
    # that these two modules can be found in bones.py and ivory/tusk.py, and that 
    # ivory/__init__.py exists as well. 
    py_modules = ['bones', 'ivory.tusk'],
        
    # Dependencies on other Python modules and packages can be specified by 
    # supplying the requires keyword argument to setup(). The value must be a 
    # list of strings. Each string specifies a package that is required, 
    # and optionally what versions are sufficient.

    # To specify that any version of a module or package is required, the string 
    # should consist entirely of the module or package name. 

    # If specific versions are required, a sequence of qualifiers can be supplied 
    # in parentheses. Each qualifier may consist of a comparison operator and a 
    # version number. The accepted comparison operators are:
    
    #  <    >    ==
    #  <=   >=   !=
    requires = ['mock'],
    
    
    # So far we have been dealing with pure and non-pure Python modules, which 
    # are usually not run by themselves but imported by scripts.
    
    # Scripts are files containing Python source code, intended to be started 
    # from the command line. Scripts don't require Distutils to do anything very 
    # complicated. The only clever feature is that if the first line of the 
    # script starts with #! and contains the word "python", the Distutils will 
    # adjust the first line to refer to the current interpreter location. By 
    # default, it is replaced with the current interpreter location. 

    # The scripts option simply is a list of files to be handled in this way. 
    scripts=['skeleton/scripts/skull', 'skeleton/scripts/cross_bones'],
        
    # Often, additional files need to be installed into a package. These files 
    # are often data that's closely related to the package's implementation, or 
    # text files containing documentation that might be of interest to 
    # programmers using the package. These files are called package data.

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.msg files found in the 'hello' package, too:
        'skeleton': ['*.dat'],
    },

    # The data_files option can be used to specify additional files needed by 
    # the module distribution: configuration files, message catalogs, data 
    # files, anything which doesn't fit in the previous categories.

    # data_files specifies a sequence of (directory, files) pairs in the 
    # following way:

    # You can specify the directory names where the data files will be 
    # installed, but you cannot rename the data files themselves.

    # Each (directory, files) pair in the sequence specifies the installation 
    # directory and the files to install there. If directory is a relative path, 
    # it is interpreted relative to the installation prefix (Python's sys.prefix 
    # for pure-Python packages, sys.exec_prefix for packages that contain 
    # extension modules). Each file name in files is interpreted relative to the 
    # setup.py script at the top of the package source distribution. No 
    # directory information from files is used to determine the final location 
    # of the installed file; only the name of the file is used.

#    entry_points = {
#        'console_scripts': ['cross_bones_too = skeleton:scripts:cross_bones:run'],
#    },
#
#    entry_points = {
#        'setuptools.file_finders': ['dummy = setuptools_dummy:dummylsfiles'],
#    },


    test_suite = "skeleton.tests.test_all",

    )

