#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Flight Data Services Ltd
# http://www.flightdataservices.com
# See the file "LICENSE" for the full license governing this code.

from setuptools import setup, find_packages
from requirements import RequirementsParser
requirements = RequirementsParser()

setup(
    long_description=open('README.rst').read() + open('CHANGES').read() + open('TODO').read() + open('AUTHORS').read(),
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements.install_requires,
    setup_requires=requirements.setup_requires,
    tests_require=requirements.tests_require,
    extras_require=requirements.extras_require,
    dependency_links=requirements.dependency_links,
    test_suite='nose.collector',
)

################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
