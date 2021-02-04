try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import sys
pymod_name = "parcels_nodes"
py_version='python%d.%d' % (sys.version_info[0],sys.version_info[1])

d = setup(name = pymod_name,
          version = '0.1',
          description ="""minimal running example of Parcels, combined with pre-compiled double-linked note lists""",
          author="oceanparcels.org + Dr. C. Kehl",
          setup_requires=['setuptools_scm',],
          packages=find_packages(),
          package_data={pymod_name: ['include/*',
                                     'src/*',
                                     'examples/*']},
)
