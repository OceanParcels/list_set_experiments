import sys
from ctypes import byref
from ctypes import c_double
from ctypes import c_float
from ctypes import c_int
# from ctypes import c_void_p
# from ctypes import POINTER
from ctypes import pointer
import numpy as np
# import _ctypes
# import numpy.ctypeslib as npct
# from copy import deepcopy

from parcels_nodes.kernelbase import BaseFieldKernel, BaseNoFieldKernel

# from codegenerator import KernelGenerator, LoopGenerator
# from compiler import get_cache_dir

# from parcels_mocks import Field

# from parcels_nodes.package_globals import get_cache_dir
from parcels_nodes.parcels_mocks import NestedField
from parcels_nodes.parcels_mocks import SummedField
from parcels_nodes.parcels_mocks import VectorField
from parcels_nodes.parcels_mocks import StatusCode as ErrorCode

import math

# from particleset_node import ParticleSet

__all__ = ['NodeNoFieldKernel', 'NodeFieldKernel']


DEBUG_MODE = False


def DoNothing(particle, fieldset, time):
    return ErrorCode.Success

def PrintKernel(particle, fieldset, time):
    print("Particle %d ..." % (particle.id))

class NodeNoFieldKernel(BaseNoFieldKernel):
    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        super(NodeNoFieldKernel, self).__init__(fieldset, ptype, pyfunc, funcname, funccode, py_ast, funcvars, c_include, delete_cfiles)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        # if len(pset.particles) > 0:
        #     assert pset.fieldset.gridset.size == len(pset.particles[0].xi), 'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
        if len(pset) > 0:
            assert pset.fieldset.gridset.size == len(pset[0].data.xi), \
                'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
        for g in pset.fieldset.gridset.grids:
        #    g.cstruct = None  # This force to point newly the grids from Python to C
            g.reset_cstruct()

        # # Make a copy of the transposed array to enforce
        # # C-contiguous memory layout for JIT mode.
        # for f in pset.fieldset.get_fields():
        #     if type(f) in [VectorField, NestedField, SummedField]:
        #         continue
        #     if f in self.field_args.values():
        #         f.chunk_data()
        #     else:
        #         for block_id in range(len(f.data_chunks)):
        #             #del f.data_chunks[block_id]
        #             f.data_chunks[block_id] = None
        #             f.c_data_chunks[block_id] = None

        # for g in pset.fieldset.gridset.grids:
        #     g.load_chunk = np.where(g.load_chunk == 1, 2, g.load_chunk)
        #     if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
        #         if not g.load_chunk.flags.c_contiguous:
        #             g.load_chunk = g.load_chunk.copy()
        #     if not g.depth.flags.c_contiguous:
        #         g.depth = g.depth.copy()
        #     if not g.lon.flags.c_contiguous:
        #         g.lon = g.lon.copy()
        #     if not g.lat.flags.c_contiguous:
        #         g.lat = g.lat.copy()

        fargs = []
        if self.field_args is not None:
            fargs += [byref(f.ctypes_struct) for f in self.field_args.values()]
        if self.const_args is not None:
            fargs += [c_double(f) for f in self.const_args.values()]

        # particle_data = pset._particle_data.ctypes.data_as(c_void_p)
        node_data = pset.begin()
        if len(fargs) > 0:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt), *fargs)
        else:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt))

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        #for f in self.fieldset.get_fields():
        #    if type(f) in [VectorField, NestedField, SummedField]:
        #        continue
        #    f.data = np.array(f.data)

        # ========= OLD ======= #
        # for p in pset.particles:
        # ===================== #
        node = pset.begin()
        while node is not None:
            p = node.data
            ptype = p.getPType()
            # Don't execute particles that aren't started yet
            sign_end_part = np.sign(endtime - p.time)
            dt_pos = min(abs(p.dt), abs(endtime - p.time))

            # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
            # as they fulfil the condition here on entering at the final calculation here. ==== #
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(p.time) >= abs(endtime):
                    p.state = ErrorCode.Success
                node = node.next
                continue

            # Compute min/max dt for first timestep
            # while dt_pos > 1e-6 or dt == 0:
            while p.state in [ErrorCode.Evaluate, ErrorCode.Repeat] or np.isclose(dt, 0):
                for var in ptype.variables:
                    p_var_back[var.name] = getattr(p, var.name)
                try:
                    pdt_prekernels = sign_dt * dt_pos
                    p.dt = pdt_prekernels
                    state_prev = p.state
                    # res = self.pyfunc(p, None, p.time)
                    res = self.pyfunc(p, pset.fieldset, p.time)
                    if res is None:
                        res = ErrorCode.Success

                    if res is ErrorCode.Success and p.state != state_prev:
                        res = p.state

                    if res == ErrorCode.Success and not np.isclose(p.dt, pdt_prekernels):
                        res = ErrorCode.Repeat

                #except FieldOutOfBoundError as fse_xy:
                #    res = ErrorCode.ErrorOutOfBounds
                #    p.exception = fse_xy
                #except FieldOutOfBoundSurfaceError as fse_z:
                #    res = ErrorCode.ErrorThroughSurface
                #    p.exception = fse_z
                #except TimeExtrapolationError as fse_t:
                #    res = ErrorCode.ErrorTimeExtrapolation
                #    p.exception = fse_t
                except Exception as e:
                    res = ErrorCode.Error
                    p.exception = e

                # Handle particle time and time loop
                if res in [ErrorCode.Success, ErrorCode.Delete]:
                    # Update time and repeat
                    p.time += p.dt
                    p.update_next_dt()
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if res != ErrorCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                        res = ErrorCode.Evaluate
                    if sign_end_part != sign_dt:
                        dt_pos = 0

                    p.state = res
                    if np.isclose(dt, 0):
                        break
                else:
                    p.state = res
                    # Try again without time update
                    for var in ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(p, var.name, p_var_back[var.name])
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if sign_end_part != sign_dt:
                        dt_pos = 0
                    break
            node = node.next

    def execute(self, pset, endtime, dt, recovery=None, output_file=None):
        """Execute this Kernel over a ParticleSet for several timesteps"""

        node = pset.begin()
        while node is not None:
            node.data.reset_state()

        def _print_error_occurred_(particle, fieldset, time):
            print("An error occurred during execution with particle={} at time={}".format(particle, time))

        # ==== this one really should be a member function of the ParticleSet, with the outfile as parameter ==== #
        def _remove_deleted_(pset, verbose=False):
            """Utility to remove all particles that signalled deletion"""
            rm_indices = pset.get_deleted_item_indices()
            if len(rm_indices) > 0:
                pdata = [pset[i].data for i in rm_indices]
                if len(pdata) > 0 and output_file is not None:
                    output_file.write(pdata, endtime, deleted_only=True)
                if DEBUG_MODE and len(pdata) > 0 and verbose:
                    sys.stdout.write("|P| before delete: {}\n".format(len(pset)))
                # pset.remove(indices)
                pset.remove_deleted_items_by_indices(rm_indices)
                if DEBUG_MODE and len(pdata) > 0 and verbose:
                    sys.stdout.write("|P| after delete: {}\n".format(len(pset)))
            return pset

        if recovery is None:
            recovery = {}
        elif ErrorCode.ErrorOutOfBounds in recovery and ErrorCode.ErrorThroughSurface not in recovery:
            recovery[ErrorCode.ErrorThroughSurface] = recovery[ErrorCode.ErrorOutOfBounds]
        # recovery_map = recovery_base_map.copy()
        recovery_map = {ErrorCode.Error: _print_error_occurred_,
                ErrorCode.ErrorInterpolation: _print_error_occurred_,
                ErrorCode.ErrorOutOfBounds: _print_error_occurred_,
                ErrorCode.ErrorTimeExtrapolation: _print_error_occurred_,
                ErrorCode.ErrorThroughSurface: _print_error_occurred_}
        recovery_map.update(recovery)

        # for g in pset.fieldset.gridset.grids:
        #     if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
        #         g.load_chunk = np.where(g.load_chunk == 2, 3, g.load_chunk)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        _remove_deleted_(pset)

        # Identify particles that threw errors
        # ====================================== #
        # ==== EXPENSIVE LIST COMPREHENSION ==== #
        # ====================================== #
        # error_particles = [p for p in pset.particles if p.state != ErrorCode.Success]
        error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]

        error_loop_iter = 0
        while len(error_particles) > 0:
            # Apply recovery kernel
            for p in error_particles:
                if p.state == ErrorCode.StopExecution:
                    return
                if p.state == ErrorCode.Repeat:
                    p.reset_state()
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.state = ErrorCode.Success
                    recovery_kernel(p, self.fieldset, p.time)
                    # recovery_kernel(p, None, p.time)
                    if(p.isComputed()):
                        p.reset_state()
                else:
                    if DEBUG_MODE:
                        sys.stdout.write("Error: loop={},  p.state={}, recovery_map: {}, age: {}, agetime: {}\n".format(error_loop_iter, p.state,recovery_map, p.age, p.agetime))
                    p.delete()

            if DEBUG_MODE:
                before_len = len(pset.particles)

            # Remove all particles that signalled deletion
            _remove_deleted_(pset)

            if DEBUG_MODE:
                after_len = len(pset.particles)
                # remaining_delete_indices = len([i for i, p in enumerate(pset.particles) if p.state in [ErrorCode.Delete]])
                remaining_delete_indices = len(pset.get_deleted_item_indices())

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            if DEBUG_MODE:
                # recalc_delete_indices = len([i for i, p in enumerate(pset.particles) if p.state in [ErrorCode.Delete]])
                recalc_delete_indices = len(pset.get_deleted_item_indices())
                if before_len != after_len:
                    sys.stdout.write("removed particles in main: {}; remaining delete particles: {}\n".format(before_len-after_len, remaining_delete_indices))
                if recalc_delete_indices > 0 or remaining_delete_indices > 0:
                    sys.stdout.write("remaining delete particles after delete(): {}; new delete particles after execute(): {}\n".format(remaining_delete_indices, recalc_delete_indices))

            # ====================================== #
            # ==== EXPENSIVE LIST COMPREHENSION ==== #
            # ====================================== #
            # error_particles = [p for p in pset.particles if p.state != ErrorCode.Success]
            error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]
            error_loop_iter += 1


class NodeFieldKernel(BaseFieldKernel):
    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        super(NodeFieldKernel, self).__init__(fieldset, ptype, pyfunc, funcname, funccode, py_ast, funcvars, c_include, delete_cfiles)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        # if len(pset.particles) > 0:
        #     assert pset.fieldset.gridset.size == len(pset.particles[0].xi), 'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
        if len(pset) > 0:
            assert pset.fieldset.gridset.size == len(pset[0].data.xi), \
                'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
        for g in pset.fieldset.gridset.grids:
        #    g.cstruct = None  # This force to point newly the grids from Python to C
            g.reset_cstruct()

        # # Make a copy of the transposed array to enforce
        # # C-contiguous memory layout for JIT mode.
        # for f in pset.fieldset.get_fields():
        #     if type(f) in [VectorField, NestedField, SummedField]:
        #         continue
        #     if f in self.field_args.values():
        #         f.chunk_data()
        #     else:
        #         for block_id in range(len(f.data_chunks)):
        #             #del f.data_chunks[block_id]
        #             f.data_chunks[block_id] = None
        #             f.c_data_chunks[block_id] = None

        # for g in pset.fieldset.gridset.grids:
        #     g.load_chunk = np.where(g.load_chunk == 1, 2, g.load_chunk)
        #     if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
        #         if not g.load_chunk.flags.c_contiguous:
        #             g.load_chunk = g.load_chunk.copy()
        #     if not g.depth.flags.c_contiguous:
        #         g.depth = g.depth.copy()
        #     if not g.lon.flags.c_contiguous:
        #         g.lon = g.lon.copy()
        #     if not g.lat.flags.c_contiguous:
        #         g.lat = g.lat.copy()

        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        # particle_data = pset._particle_data.ctypes.data_as(c_void_p)
        node_data = pset.begin()
        self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt), *fargs)

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        for f in self.fieldset.get_fields():
            if type(f) in [VectorField, NestedField, SummedField]:
                continue
            f.data = np.array(f.data)

        # ========= OLD ======= #
        # for p in pset.particles:
        # ===================== #
        node = pset.begin()
        while node is not None:
            p = node.data
            ptype = p.getPType()
            # Don't execute particles that aren't started yet
            sign_end_part = np.sign(endtime - p.time)
            dt_pos = min(abs(p.dt), abs(endtime - p.time))

            # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
            # as they fulfil the condition here on entering at the final calculation here. ==== #
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(p.time) >= abs(endtime):
                    p.state = ErrorCode.Success
                node = node.next
                continue

            # Compute min/max dt for first timestep
            # while dt_pos > 1e-6 or dt == 0:
            while p.state in [ErrorCode.Evaluate, ErrorCode.Repeat] or np.isclose(dt, 0):
                for var in ptype.variables:
                    p_var_back[var.name] = getattr(p, var.name)
                try:
                    pdt_prekernels = sign_dt * dt_pos
                    p.dt = pdt_prekernels
                    state_prev = p.state
                    # res = self.pyfunc(p, None, p.time)
                    res = self.pyfunc(p, pset.fieldset, p.time)
                    if res is None:
                        res = ErrorCode.Success

                    if res is ErrorCode.Success and p.state != state_prev:
                        res = p.state

                    if res == ErrorCode.Success and not np.isclose(p.dt, pdt_prekernels):
                        res = ErrorCode.Repeat

                #except FieldOutOfBoundError as fse_xy:
                #    res = ErrorCode.ErrorOutOfBounds
                #    p.exception = fse_xy
                #except FieldOutOfBoundSurfaceError as fse_z:
                #    res = ErrorCode.ErrorThroughSurface
                #    p.exception = fse_z
                #except TimeExtrapolationError as fse_t:
                #    res = ErrorCode.ErrorTimeExtrapolation
                #    p.exception = fse_t
                except Exception as e:
                    res = ErrorCode.Error
                    p.exception = e

                # Handle particle time and time loop
                if res in [ErrorCode.Success, ErrorCode.Delete]:
                    # Update time and repeat
                    p.time += p.dt
                    p.update_next_dt()
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if res != ErrorCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                        res = ErrorCode.Evaluate
                    if sign_end_part != sign_dt:
                        dt_pos = 0

                    p.state = res
                    if np.isclose(dt, 0):
                        break
                else:
                    p.state = res
                    # Try again without time update
                    for var in ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(p, var.name, p_var_back[var.name])
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if sign_end_part != sign_dt:
                        dt_pos = 0
                    break
            node = node.next

    def execute(self, pset, endtime, dt, recovery=None, output_file=None):
        """Execute this Kernel over a ParticleSet for several timesteps"""

        node = pset.begin()
        while node is not None:
            node.data.reset_state()

        def _print_error_occurred_(particle, fieldset, time):
            print("An error occurred during execution with particle={} at time={}".format(particle, time))


        def _remove_deleted_(pset, verbose=False):
            """Utility to remove all particles that signalled deletion"""
            rm_indices = pset.get_deleted_item_indices()
            if len(rm_indices) > 0:
                pdata = [pset[i].data for i in rm_indices]
                if len(pdata) > 0 and output_file is not None:
                    output_file.write(pdata, endtime, deleted_only=True)
                if DEBUG_MODE and len(pdata) > 0 and verbose:
                    sys.stdout.write("|P| before delete: {}\n".format(len(pset)))
                # pset.remove(indices)
                pset.remove_deleted_items_by_indices(rm_indices)
                if DEBUG_MODE and len(pdata) > 0 and verbose:
                    sys.stdout.write("|P| after delete: {}\n".format(len(pset)))
            return pset

        if recovery is None:
            recovery = {}
        elif ErrorCode.ErrorOutOfBounds in recovery and ErrorCode.ErrorThroughSurface not in recovery:
            recovery[ErrorCode.ErrorThroughSurface] = recovery[ErrorCode.ErrorOutOfBounds]
        # recovery_map = recovery_base_map.copy()
        recovery_map = {ErrorCode.Error: _print_error_occurred_,
                ErrorCode.ErrorInterpolation: _print_error_occurred_,
                ErrorCode.ErrorOutOfBounds: _print_error_occurred_,
                ErrorCode.ErrorTimeExtrapolation: _print_error_occurred_,
                ErrorCode.ErrorThroughSurface: _print_error_occurred_}
        recovery_map.update(recovery)

        # for g in pset.fieldset.gridset.grids:
        #     if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
        #         g.load_chunk = np.where(g.load_chunk == 2, 3, g.load_chunk)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        _remove_deleted_(pset)

        # Identify particles that threw errors
        # ====================================== #
        # ==== EXPENSIVE LIST COMPREHENSION ==== #
        # ====================================== #
        # error_particles = [p for p in pset.particles if p.state != ErrorCode.Success]
        error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]

        error_loop_iter = 0
        while len(error_particles) > 0:
            # Apply recovery kernel
            for p in error_particles:
                if p.state == ErrorCode.StopExecution:
                    return
                if p.state == ErrorCode.Repeat:
                    p.reset_state()
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.state = ErrorCode.Success
                    recovery_kernel(p, self.fieldset, p.time)
                    # recovery_kernel(p, None, p.time)
                    if(p.isComputed()):
                        p.reset_state()
                else:
                    if DEBUG_MODE:
                        sys.stdout.write("Error: loop={},  p.state={}, recovery_map: {}, age: {}, agetime: {}\n".format(error_loop_iter, p.state,recovery_map, p.age, p.agetime))
                    p.delete()

            if DEBUG_MODE:
                before_len = len(pset.particles)

            # Remove all particles that signalled deletion
            _remove_deleted_(pset)

            if DEBUG_MODE:
                after_len = len(pset.particles)
                # remaining_delete_indices = len([i for i, p in enumerate(pset.particles) if p.state in [ErrorCode.Delete]])
                remaining_delete_indices = len(pset.get_deleted_item_indices())

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            if DEBUG_MODE:
                # recalc_delete_indices = len([i for i, p in enumerate(pset.particles) if p.state in [ErrorCode.Delete]])
                recalc_delete_indices = len(pset.get_deleted_item_indices())
                if before_len != after_len:
                    sys.stdout.write("removed particles in main: {}; remaining delete particles: {}\n".format(before_len-after_len, remaining_delete_indices))
                if recalc_delete_indices > 0 or remaining_delete_indices > 0:
                    sys.stdout.write("remaining delete particles after delete(): {}; new delete particles after execute(): {}\n".format(remaining_delete_indices, recalc_delete_indices))

            # ====================================== #
            # ==== EXPENSIVE LIST COMPREHENSION ==== #
            # ====================================== #
            # error_particles = [p for p in pset.particles if p.state != ErrorCode.Success]
            error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]
            error_loop_iter += 1



