#!/usr/bin/env python

# The purpose of Skeleton is to serve as an example of a basic Python package
# including SetupTools capabilities. Comments have been adapted from
#
#  - http://docs.python.org/distutils/
#  - http://pypi.python.org/pypi/setuptools
#  - http://peak.telecommunity.com/DevCenter/setuptools
#  - http://ianbicking.org/docs/setuptools-presentation/
#  - http://github.com/ella/setuptools-dummy
#
# TODO
#  - Cython extension modules
#  `--- http://docs.cython.org/docs/tutorial.html
#  - Platform specific requirements, Windows vs Linux vs Mac OS X
#  `--- http://tarekziade.wordpress.com/2009/09/12/static-metadata-for-distutils/
#  - pip compatibility
#  `--- http://pypi.python.org/pypi/pip/
#  - Distribute compatibility
#  `--- http://pypi.python.org/pypi/distribute

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
    # === Meta data ===
    
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

    # === Include and Exclude ===

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
      
    packages = find_packages(exclude=['ez_setup']),     
                
    # Often, additional files need to be installed into a package. These files 
    # are often data that's closely related to the package's implementation, or 
    # text files containing documentation that might be of interest to 
    # programmers using the package. These files are called package data.

    # Setuptools offers three ways to specify data files to be included in your 
    # packages. First, you can simply use the include_package_data keyword. This 
    # tells setuptools to install any data files it finds in your packages. The 
    # data files must be under CVS or Subversion control, or else they must be 
    # specified via the distutils' MANIFEST.in file. (They can also be tracked 
    # by another revision control system, using an appropriate plugin. 

    #include_package_data = True, 

    # If you want finer-grained control over what files are included (for 
    # example, if you have documentation files in your package directories and 
    # want to exclude them from installation), then you can also use the 
    # package_data keyword, e.g.:

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
        # And include any *.dat files found in the 'data' directory of the 
        # 'skeleton' package amd all the files in the 'scripts' directory
        # of the 'skeleton' package too.
        'skeleton': ['data/*.dat'],
        'skeleton': ['scripts/*'],
    },
        
    # === Dependancies ===        
        
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
    requires = ['configobj'],


    # A string or list of strings specifying what other distributions need to be 
    # installed when this one is.
    install_requires = ['cython'],
         
    # Sometimes a project has "recommended" dependencies, that are not required 
    # for all uses of the project. For example, a project might offer optional 
    # reStructuredText support if docutils is installed. These optional features 
    # are called "extras", and setuptools allows you to define their requirements 
    # as well. In this way, other projects that require these optional features 
    # can force the additional requirements to be installed, by naming the 
    # desired extras in their install_requires.
      
    # A dictionary mapping names of "extras" (optional features of your project) 
    # to strings or lists of strings specifying what other distributions must be 
    # installed to support those features.    
    extras_require = {
        'reST': ["docutils>=0.3"],
    },

    # If your project depends on packages that aren't registered in PyPI, you 
    # may still be able to depend on them, as long as they are available for 
    # download as an egg, in the standard distutils sdist format, or as a single 
    # .py file. You just need to add some URLs to the dependency_links argument 
    # to setup().

    # The URLs must be either:
    #  - direct download URLs, or
    #  - the URLs of web pages that contain direct download links

    # In general, it's better to link to web pages, because it is usually less 
    # complex to update a web page than to release a new version of your 
    # project. You can also use a SourceForge showfiles.php link in the case 
    # where a package you depend on is distributed via SourceForge.

    # The dependency_links option takes the form of a list of URL strings. 
    # For example, the below will cause EasyInstall to search the specified page 
    # for eggs or source distributions, if the package's dependencies aren't 
    # already installed:

    dependency_links = [
        'http://vindictive.flightdataservices.com/Trac/Nelson/snapshots/'
    ],

    # === Script Creation ===
    
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

    # === Entry Points (are better than 'scripts') ===

    # Packaging and installing scripts can be a bit awkward using the method 
    # above For one thing, there's no easy way to have a script's filename match 
    # local conventions on both Windows and POSIX platforms. For another, you 
    # often have to create a separate file just for the "main" script, when your 
    # actual "main" is a function in a module somewhere. 

    # setuptools fixes all of these problems by automatically generating scripts 
    # for you with the correct extension, and on Windows it will even create an 
    # .exe file so that users don't have to change their PATHEXT settings. The 
    # way to use this feature is to define "entry points" in your setup script 
    # that indicate what function the generated script should import and run. 
    # It is possible to create console scripts and GUI scripts.        
        
    # Two notations are popular, the first is easy to read and maintain.
    
    entry_points = """
        [console_scripts]
        cross_bones_too = skeleton.cross_bones_too:run
        skull_too = skeleton.skull_too:run
        
        #[gui_scripts]
        #skull_gui = skeleton.gui.skull_gui:run
        """,
        
    #entry_points = {
    #    'console_scripts': [
    #        'cross_bones_too = skeleton.scripts.cross_bones_too:run',
    #        'skull_too = skeleton.scripts.skull_too:run',
    #    ],
    #    'gui_scripts': [
    #        'skull_gui = skeleton.gui.skull_gui:run',
    #    ]
    #},

    # A string naming a unittest.TestCase subclass (or a package or module 
    # containing one or more of them, or a method of such a subclass), or naming 
    # a function that can be called with no arguments and returns a 
    # unittest.TestSuite. If the named suite is a module, and the module has an 
    # additional_tests() function, it is called and the results are added to the 
    # tests to be run. If the named suite is a package, any submodules and 
    # subpackages are recursively added to the overall test suite.

    # Specifying this argument enables use of the test command to run the 
    # specified test suite, e.g. via setup.py test. See the section on the test 
    # command below for more details.

    #test_suite = "skeleton.tests, ivory.test",
        
    # If your project's tests need one or more additional packages besides those 
    # needed to install it, you can use this option to specify them. It should 
    # be a string or list of strings specifying what other distributions need to 
    # be present for the package's tests to run. 
    
    test_requires = ['mock'],

    # A boolean flag specifying whether the project can be safely installed and 
    # run from a zip file. If this argument is not supplied, the bdist_egg 
    # command will have to analyze all of your project's contents for possible 
    # problems each time it buids an egg. 
    zip_safe = False,
    
    )
