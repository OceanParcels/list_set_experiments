import functools
import sys
from ctypes import c_double
from ctypes import c_float
from ctypes import c_int
from ctypes import c_void_p
from ctypes import cast
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure
from enum import IntEnum
import numpy as np

__all__ = ['CGrid', 'Grid', 'GridSet']


class CGrid(Structure):
    _fields_ = [('gtype', c_int),
                ('grid', c_void_p)]


class Grid(object):
    _cstruct = None

    def __init__(self, time, xdim, ydim, zdim, tdim):
        self.gtype = 1
        # self.depth
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.tdim = tdim
        self.time = time
        self._cstruct = None
        self._initialized = False

    @property
    def ctypes_struct(self):
        # This is unnecessary for the moment, but it could be useful when going will fully unstructured grids
        self.cgrid = cast(pointer(self.child_ctypes_struct), c_void_p)
        cstruct = CGrid(self.gtype, self.cgrid.value)
        return cstruct

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CStructuredGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int),
                        # ('depth', POINTER(c_float)),
                        ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self._cstruct:  # Not to point to the same grid various times if grid in various fields
            if not isinstance(self.periods, c_int):
                self.periods = c_int()
                self.periods.value = 0
            self._cstruct = CStructuredGrid(self.xdim, self.ydim, self.zdim,
                                           self.tdim,
                                           # self.depth.ctypes.data_as(POINTER(c_float)),
                                           self.time.ctypes.data_as(POINTER(c_double)))
        return self._cstruct

    @property
    def cstruct(self):
        return self._cstruct

    def reset_cstruct(self):
        self._cstruct = None

    def computeTimeChunk(self, field, time_value, signdt):
        time_indices_now = np.array([False,])
        time_indices_next = np.array([False, ])
        time_now = 0
        time_next = 0
        if signdt >= 0:
            time_indices_lower = self.time<time_value
            time_indices_higher = self.time>time_value
            # ==== NOW ==== #
            time_indices_lower_ceil = np.concatenate((np.array([True,]), time_indices_lower[1:]))
            time_indices_higher_int = np.concatenate((time_indices_higher[1:], np.array([True,])))
            time_indices_now = np.logical_and(time_indices_lower_ceil, time_indices_higher_int)
            # ==== NEXT ==== #
            time_indices_higher_int = np.concatenate((np.array([True,]), time_indices_lower[:-1]))
            time_indices_higher_ceil = np.concatenate((time_indices_higher[:-1], np.array([True,])))
            time_indices_next = np.logical_and(time_indices_higher_int, time_indices_higher_ceil)
        else:
            time_indices_lower = self.time>time_value
            time_indices_higher = self.time<time_value
            # ==== NOW ==== #
            time_indices_lower_int = np.concatenate((time_indices_lower[:-1], np.array([True,])))
            time_indices_higher_ceil = np.concatenate((np.array([True,]), time_indices_higher[:-1]))
            time_indices_now = np.logical_and(time_indices_lower_int, time_indices_higher_ceil)
            # ==== NEXT ==== #
            time_indices_lower_int = np.concatenate((np.array([True,]), time_indices_higher[1:]))
            time_indices_lower_ceil = np.concatenate((time_indices_lower[1:], np.array([True,])))
            time_indices_next = np.logical_and(time_indices_lower_int, time_indices_lower_ceil)
        time_now = self.time[time_indices_now]
        time_next = self.time[time_indices_next]
        #if not self._initialized:
        #    self._initialized = True
        #    return time_now
        return time_next




class GridSet(object):
    time = []
    ti = 0

    def __init__(self):
        self._data = []
        self.time = []
        self.ti = 0

    def __del__(self):
        del self._data[:]
        self._data = None

    @property
    def grids(self):
        return self._data

    def append(self, obj):
        self._data.append(obj)

    def set_time_by_minmax(self, min_time, max_time, grid_dt):
        self.time = np.arange(min_time, max_time, grid_dt, dtype=np.float64)

    def set_time_by_numpy(self, time):
        self.time = time

    def dimrange(self, var_name):
        return self.time[0], self.time[-1]

    def computeTimeChunk(self, field, time_value, sign_dt=1):
        """
        computes the next-step time index
        :param field: a connected field for which the index is being computed
        :param time_value: requested time value in the field
        :param sign_dt: sign of the timeline - >0 = forward, <0 = backward
        :return:
        """
        pass


    @property
    def size(self):
        return len(self._data)

    def __len__(self):
        return len(self._data)
