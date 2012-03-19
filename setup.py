#!/usr/bin/env python
# -*- coding: utf-8 -*-

# An interface for HDF files containing flight data.
# Copyright (c) 2009-2012 Flight Data Services Ltd
# http://www.flightdataservices.com

try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

from hdfaccess import __version__ as VERSION
from requirements import RequirementsParser
requirements = RequirementsParser()

setup(
    name='HDFAccess',
    version=VERSION,   
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',
    description='An interface for HDF files containing flight data.',    
    long_description=open('README').read() + open('CHANGES').read(),
    license='Other/Proprietary License',
    url='http://www.flightdatacommunity.com/',
    download_url='',    
    packages=find_packages(exclude=("tests",)),
    # The 'include_package_data' keyword tells setuptools to install any 
    # data files it finds specified in the MANIFEST.in file.    
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements.install_requires,
    setup_requires=requirements.setup_requires,
    tests_require=requirements.tests_require,
    extras_require=requirements.extras_require,
    dependency_links=requirements.dependency_links,
    test_suite='nose.collector',
    platforms=[
        'OS Independent',
    ],        
    keywords=['hdf', 'numpy', 'flight', 'data'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Operating System :: OS Independent',
        'Topic :: Software Development',
    ],
)
