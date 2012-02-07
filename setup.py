#!/usr/bin/env python

import re

try:
    from setuptools import setup, find_packages
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

def parse_requirements(file_name):
    """
    Extract all dependency names from requirements.txt.
    """
    requirements = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'(\s*#)|(\s*$)', line):
            continue
        if re.match(r'\s*-e\s+', line):
            # TODO support version numbers
            requirements.append(re.sub(r'\s*-e\s+.*#egg=(.*)$', r'\1', line))
        elif re.match(r'\s*-f\s+', line):
            pass
        else:
            requirements.append(line)

    requirements.reverse()
    return requirements

def parse_dependency_links(file_name):
    """
    Extract all URLs for packages not found on PyPI.
    """
    dependency_links = []
    for line in open(file_name, 'r').read().split('\n'):
        if re.match(r'\s*-[ef]\s+', line):
            dependency_links.append(re.sub(r'\s*-[ef]\s+', '', line))

    dependency_links.reverse()
    return dependency_links

from hdfaccess import __version__ as VERSION

setup(
    name='HDFAccess',
    version = VERSION,
    url='http://www.flightdataservices.com/',
    author='Flight Data Services Ltd',
    author_email='developers@flightdataservices.com',
    description='An interface for HDF files containing flight data.',
    long_description = open('README').read() + open('CHANGES').read(),
    download_url='',
    platforms='',
    license='License :: OSI Approved :: Open Software License (OSL)',

    packages = find_packages(),
    include_package_data = True,

    # Parse the 'requirements.txt' file to determine the dependencies.
    install_requires = parse_requirements('requirements.txt'),
    dependency_links = parse_dependency_links('requirements.txt'),
    setup_requires = ['nose>=1.0'],
    test_suite = 'nose.collector',
    zip_safe = False,
    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Open Software License (OSL)",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Operating System :: OS Independent",
        "Topic :: Utilities",
        ],

    )