try:
    from setuptools import setup, find_packages, Extension, Distribution
except ImportError:
    from distutils.core import setup, find_packages, Extension
    from distutils.dist import Distribution

import os
import sys
# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


# Get our own instance of Distribution
dist = Distribution()
dist.parse_config_files()
dist.parse_command_line()

# Get prefix from either config file or command line
prefix = dist.get_option_dict('install')['prefix'][1]
print("Prefix is: " + prefix)

pymod_name = "parcels_nodes"
py_version='python%d.%d' % (sys.version_info[0],sys.version_info[1])
print(sys.prefix)
PCLS_NODES_HOME='/git_code/list_set_experiments'  # '/vps/otbknox/williams/OTB_2.0'
PCLS_NODES_INCLDIR=[
      os.path.join(PCLS_NODES_HOME, pymod_name, 'include'),
      os.path.join(sys.prefix, 'include'),
      os.path.join(sys.prefix,'include', py_version),
      os.path.join(sys.prefix,'lib',py_version,'site-packages',pymod_name,'include'),
      os.path.join(prefix, py_version,'site-packages',pymod_name, 'include')
    ]
PCLS_NODES_LIBDIR=[os.path.join(PCLS_NODES_HOME,'lib'),
                   os.path.join(prefix,'lib',py_version,'site-packages',pymod_name),
                   os.path.join(prefix,'lib',py_version,'site-packages',pymod_name,'lib'),]

# nodes_c_module = Extension('node', sources = ['node.c'])
nodes_swig_module = Extension('_node',
                              sources=["%s/wrapping/node.i" % (pymod_name),"%s/src/node.c" % (pymod_name)], swig_opts=['-modern'],
                              include_dirs=["%s/include" % (pymod_name),]+PCLS_NODES_INCLDIR+[numpy_include,],
                              library_dirs=["."]+PCLS_NODES_LIBDIR,
                              # libraries=['node'],
                              )  #, '-includeall' , '-I.'

d = setup(name = pymod_name,
          version = '0.1',
          description ="""minimal running example of Parcels, combined with pre-compiled double-linked note lists""",
          author="oceanparcels.org + Dr. C. Kehl",
          setup_requires=['setuptools_scm',],
          packages=find_packages(),
          package_data={pymod_name: ['include/*',
                                     'src/*',
                                     'examples/*']},
          # ext_modules = [nodes_c_module,]),
          ext_modules = [nodes_swig_module,],
          # ext_modules = [nodes_c_module,nodes_swig_module,],
          # py_modules = ['parcels_nodes.node',],
          # entry_points={'console_scripts': [
          #     'parcels_get_examples = parcels.scripts.get_examples:main',
          #     'parcels_convert_npydir_to_netcdf = parcels.scripts.convert_npydir_to_netcdf:main']},
)
# print(d.get_option_dict('install'))
