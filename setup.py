#!/usr/bin/env python

# The purpose of Skeleton is to serve as an example of a basic Python package
# including DistUtils capabilities. Comments have been adapted from
#
#  - http://docs.python.org/distutils/
#  - http://docs.cython.org/docs/tutorial.html
#  - http://tarekziade.wordpress.com/2009/09/12/static-metadata-for-distutils/
#
# I have deliberatly not document everything, only the best practices and 
# features the we require. Refer to the URL above for the complete DistUtils
# documentation.

# TODO
#  - Cython extension modules
#  - Platform specific requirements

from distutils.core import setup
import sys

# Sometimes things go wrong, and the setup script doesn’t do what the developer 
# wants.
#
# For this purpose, the DISTUTILS_DEBUG environment variable can be set to 
# anything except an empty string, and distutils will now print detailed 
# information what it is doing, and prints the full traceback in case an 
# exception occurs.

DISTUTILS_DEBUG = ''

setup(
    # Required meta data
    name='Skeleton',
    version='0.1.0',
    url='http://www.flightdataservices.com/',
    
    # Optional meta data   
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',            
    description='A Skeleton Python Package with DistUtils script',    
    long_decription='A Skeleton Python Package with DistUtils script',    
    download_utl='http://www.flightdataservices.com/download/',
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
    # rather than listing packages—especially the case of a single module that 
    # goes in the “root package” (i.e., no package at all). 

    # This describes two modules, one of them in the “root” package, the other 
    # in the ivory package. Again, the default package/directory layout implies 
    # that these two modules can be found in bones.py and ivory/tusk.py, and that 
    # ivory/__init__.py exists as well. 
    py_modules = ['bones', 'ivory.tusk']
        
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
    requires = ['mock']
    
    
    # So far we have been dealing with pure and non-pure Python modules, which 
    # are usually not run by themselves but imported by scripts.
    
    # Scripts are files containing Python source code, intended to be started 
    # from the command line. Scripts don’t require Distutils to do anything very 
    # complicated. The only clever feature is that if the first line of the 
    # script starts with #! and contains the word “python”, the Distutils will 
    # adjust the first line to refer to the current interpreter location. By 
    # default, it is replaced with the current interpreter location. 

    # The scripts option simply is a list of files to be handled in this way. 
    scripts=['scripts/skull', 'scripts/cross_bones']



    
    )

