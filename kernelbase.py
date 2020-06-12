import _ctypes
import inspect
import re
import time
from ast import FunctionDef
from ast import Module
from ast import parse
from copy import deepcopy
# from ctypes import byref
# from ctypes import c_double
# from ctypes import c_float
# from ctypes import c_int
# from ctypes import c_void_p
from hashlib import md5
from os import path
from os import remove
from sys import platform
from sys import version_info

import numpy as np
import numpy.ctypeslib as npct

#from memory_profiler import profile
try:
    from mpi4py import MPI
except:
    MPI = None

# from abc import ABC

# from parcels.codegenerator import KernelGenerator
# from parcels.codegenerator import LoopGenerator
#from parcels.compiler import get_cache_dir
#from parcels.field import Field
#from parcels.field import FieldOutOfBoundError
#from parcels.field import FieldOutOfBoundSurfaceError
#from parcels.field import NestedField
#from parcels.field import SummedField
#from parcels.field import VectorField
#from parcels.kernels.advection import AdvectionRK4_3D
#from parcels.tools.loggers import logger
#from parcels.tools.error import ErrorCode

from codegenerator import KernelGenerator, LoopGenerator, NodeLoopGenerator
from compiler import get_cache_dir
# from parcels_mocks import Field
# from parcels_mocks import NestedField
# from parcels_mocks import SummedField
# from parcels_mocks import VectorField
# from parcels_mocks import StatusCode as ErrorCode

re_indent = re.compile(r"^(\s+)")

def fix_indentation(string):
    """Fix indentation to allow in-lined kernel definitions"""
    lines = string.split('\n')
    indent = re_indent.match(lines[0])
    if indent:
        lines = [l.replace(indent.groups()[0], '', 1) for l in lines]
    return "\n".join(lines)


class BaseKernel(object):
    """Base super class for base Kernel objects that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information (possibly None)
    :arg ptype: PType object for the kernel particle
    :arg pyfunc: (aggregated) Kernel function
    :arg funcname: function name
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        self.fieldset = fieldset
        self.field_args = None
        self.const_args = None
        self.ptype = ptype
        self._lib = None
        self.delete_cfiles = delete_cfiles

        # Derive meta information from pyfunc, if not given
        self.funcname = funcname or pyfunc.__name__
        self.name = "%s%s" % (ptype.name, self.funcname)
        self.ccode = ""
        self.funcvars = funcvars
        self.funccode = funccode
        self.py_ast = py_ast
        self.src_file = None
        self.lib_file = None
        self.log_file = None

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                basename = path.join(get_cache_dir(), self._cache_key) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                basename = path.join(get_cache_dir(), "%s_0" % self._cache_key)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        if self._lib is not None:
            _ctypes.FreeLibrary(self._lib._handle) if platform == 'win32' else _ctypes.dlclose(self._lib._handle)
            del self._lib
            self._lib = None
            if path.isfile(self.lib_file) and self.delete_cfiles:
                [remove(s) for s in [self.src_file, self.lib_file, self.log_file]]
        self.fieldset = None
        self.field_args = None
        self.const_args = None
        self.funcvars = None
        self.funccode = None

    @property
    def _cache_key(self):
        field_keys = ""
        if self.field_args is not None:
            field_keys = "-".join(
                ["%s:%s" % (name, field.units.__class__.__name__) for name, field in self.field_args.items()])
        key = self.name + self.ptype._cache_key + field_keys + ('TIME:%f' % time.time())
        return md5(key.encode('utf-8')).hexdigest()

    def remove_lib(self):
        # Unload the currently loaded dynamic linked library to be secure
        if self._lib is not None:
            _ctypes.FreeLibrary(self._lib._handle) if platform == 'win32' else _ctypes.dlclose(self._lib._handle)
            del self._lib
            self._lib = None
        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if path.isfile(self.lib_file):
            [remove(s) for s in [self.src_file, self.lib_file, self.log_file]]
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                basename = path.join(get_cache_dir(), self._cache_key) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                basename = path.join(get_cache_dir(), "%s_0" % self._cache_key)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        with open(self.src_file, 'w') as f:
            f.write(self.ccode)
        compiler.compile(self.src_file, self.lib_file, self.log_file)
        # logger.info("Compiled %s ==> %s" % (self.name, self.lib_file))

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        self._function = self._lib.particle_loop

    def merge(self, kernel, kclass):
        funcname = self.funcname + kernel.funcname
        func_ast = None
        if self.py_ast is not None:
            func_ast = FunctionDef(name=funcname, args=self.py_ast.args, body=self.py_ast.body + kernel.py_ast.body,
                                   decorator_list=[], lineno=1, col_offset=0)
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return kclass(self.fieldset, self.ptype, pyfunc=None,
                      funcname=funcname, funccode=self.funccode + kernel.funccode,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars,
                      delete_cfiles=delete_cfiles)

    def __add__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseKernel)

    def execute_jit(self, pset, endtime, dt):
        pass

    def execute_python(self, pset, endtime, dt):
        pass

    def execute(self, pset, endtime, dt, recovery=None, output_file=None):
        pass


class BaseNoFieldKernel(BaseKernel):
    """Base super class for Kernel objects that encapsulates auto-generated code and NEGLECTS fieldsets (for testing purposes mainly).

    :arg fieldset: FieldSet object providing the field information
    :arg ptype: PType object for the kernel particle
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(BaseNoFieldKernel, self).__init__(fieldset, ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

        # ====== TO BE DONE - IN SPECIFIC SUBCLASSES ====== #
        #if pyfunc is AdvectionRK4_3D:   # would be better if the idea of a Kernel being '2D', '3D, '4D' or 'uncertain' is captured as Attribute or as class stucture
        #    warning = False
        #    if isinstance(fieldset.W, Field) and fieldset.W.creation_log != 'from_nemo' and \
        #            fieldset.W._scaling_factor is not None and fieldset.W._scaling_factor > 0:
        #        warning = True
        #    if type(fieldset.W) in [SummedField, NestedField]:
        #        for f in fieldset.W:
        #            if f.creation_log != 'from_nemo' and f._scaling_factor is not None and f._scaling_factor > 0:
        #                warning = True
        #    if warning:
        #        logger.warning_once(
        #            'Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z.\n'
        #            '         If z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)')
        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, '__code__'):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        # Parse AST if it is not provided explicitly
        self.py_ast = py_ast or parse(fix_indentation(self.funccode)).body[0]
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx['math'] = globals()['math']
                user_ctx['random'] = globals()['random']
                user_ctx['ErrorCode'] = globals()['ErrorCode']
            except:
                # logger.warning("Could not access user context when merging kernels")
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = Module(body=[self.py_ast])
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self.pyfunc = user_ctx[self.funcname]
        else:
            self.pyfunc = pyfunc

        if version_info[0] < 3:
            numkernelargs = len(inspect.getargspec(self.pyfunc).args)
        else:
            numkernelargs = len(inspect.getfullargspec(self.pyfunc).args)

        assert numkernelargs == 3, \
            'Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon.'

        self.name = "%s%s" % (ptype.name, self.funcname)

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            # kernelgen = KernelGenerator(ptypeself.fieldset)
            # kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            # self.field_args = kernelgen.field_args
            # self.vector_field_args = kernelgen.vector_field_args
            # for f in self.vector_field_args.values():
            #     Wname = f.W.ccode_name if f.W else 'not_defined'
            #     for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ['U', 'V', 'W']):
            #         if sF_name not in self.field_args:
            #             if sF_name != 'not_defined':
            #                 self.field_args[sF_name] = getattr(f, sF_component)
            # self.const_args = kernelgen.const_args
            # loopgen = LoopGenerator(ptype, self.fieldset)

            kernelgen = KernelGenerator(ptype)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            # loopgen = LoopGenerator(ptype)
            loopgen = NodeLoopGenerator(ptype)

            if path.isfile(c_include):
                with open(c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = c_include
            # self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args, kernel_ccode, c_include_str)
            self.ccode = loopgen.generate(self.funcname, None, None, kernel_ccode, c_include_str)
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                filename = "lib"+self._cache_key
                basename = path.join(get_cache_dir(), filename) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                filename = "lib"+self._cache_key
                basename = path.join(get_cache_dir(), "%s_0" % filename)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        super(BaseNoFieldKernel, self).__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, BaseNoFieldKernel):
            kernel = BaseNoFieldKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseNoFieldKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseNoFieldKernel):
            kernel = BaseNoFieldKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseNoFieldKernel)




class BaseFieldKernel(BaseKernel):
    """Base super class for Kernel objects that encapsulates auto-generated code and that REQUIRES fieldsets.

    :arg fieldset: FieldSet object providing the field information
    :arg ptype: PType object for the kernel particle
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(BaseFieldKernel, self).__init__(fieldset, ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

        # ====== TO BE DONE - IN SPECIFIC SUBCLASSES ====== #
        #if pyfunc is AdvectionRK4_3D:   # would be better if the idea of a Kernel being '2D', '3D, '4D' or 'uncertain' is captured as Attribute or as class stucture
        #    warning = False
        #    if isinstance(fieldset.W, Field) and fieldset.W.creation_log != 'from_nemo' and \
        #            fieldset.W._scaling_factor is not None and fieldset.W._scaling_factor > 0:
        #        warning = True
        #    if type(fieldset.W) in [SummedField, NestedField]:
        #        for f in fieldset.W:
        #            if f.creation_log != 'from_nemo' and f._scaling_factor is not None and f._scaling_factor > 0:
        #                warning = True
        #    if warning:
        #        logger.warning_once(
        #            'Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z.\n'
        #            '         If z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)')
        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, '__code__'):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        # Parse AST if it is not provided explicitly
        self.py_ast = py_ast or parse(fix_indentation(self.funccode)).body[0]
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx['math'] = globals()['math']
                user_ctx['random'] = globals()['random']
                user_ctx['ErrorCode'] = globals()['ErrorCode']
            except:
                # logger.warning("Could not access user context when merging kernels")
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = Module(body=[self.py_ast])
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self.pyfunc = user_ctx[self.funcname]
        else:
            self.pyfunc = pyfunc

        if version_info[0] < 3:
            numkernelargs = len(inspect.getargspec(self.pyfunc).args)
        else:
            numkernelargs = len(inspect.getfullargspec(self.pyfunc).args)

        assert numkernelargs == 3, \
            'Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon.'

        self.name = "%s%s" % (ptype.name, self.funcname)

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(ptype, self.fieldset)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            self.field_args = kernelgen.field_args
            self.vector_field_args = kernelgen.vector_field_args
            for f in self.vector_field_args.values():
                Wname = f.W.ccode_name if f.W else 'not_defined'
                for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ['U', 'V', 'W']):
                    if sF_name not in self.field_args:
                        if sF_name != 'not_defined':
                            self.field_args[sF_name] = getattr(f, sF_component)
            self.const_args = kernelgen.const_args

            # loopgen = LoopGenerator(ptype)
            loopgen = NodeLoopGenerator(ptype)

            if path.isfile(c_include):
                with open(c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args,
                                          kernel_ccode, c_include_str)
            if MPI:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                filename = "lib"+self._cache_key
                basename = path.join(get_cache_dir(), filename) if mpi_rank == 0 else None
                basename = mpi_comm.bcast(basename, root=0)
                basename = basename + "_%d" % mpi_rank
            else:
                filename = "lib"+self._cache_key
                basename = path.join(get_cache_dir(), "%s_0" % filename)

            self.src_file = "%s.c" % basename
            self.lib_file = "%s.%s" % (basename, 'dll' if platform == 'win32' else 'so')
            self.log_file = "%s.log" % basename

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        super(BaseFieldKernel, self).__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, BaseFieldKernel):
            kernel = BaseFieldKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseFieldKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseFieldKernel):
            kernel = BaseFieldKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseFieldKernel)


