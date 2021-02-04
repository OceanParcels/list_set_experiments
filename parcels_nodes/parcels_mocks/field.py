from .grid import *
from ctypes import c_float
from ctypes import c_int
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure
import numpy as np

from datetime import timedelta


__all__ = ['Field', 'VectorField', 'SummedField', 'NestedField', 'FieldSet']


class Field(object):
    data = None

    def __init__(self, fieldset=None, time=None, name="", grid=None):
        self.fieldset = fieldset
        self.name = name
        self.data = np.empty((2, 2, 2), dtype=np.float32)
        self.c_data = self.data.flatten(order='C')
        self.c_data_ptr = None
        self.time = np.arange(0, timedelta(days=365).total_seconds(), timedelta(days=10).total_seconds(), dtype=np.float64) if time is None else time
        self.grid = Grid(self.time, 2, 2, 2, self.time.shape[0]) if grid is None else grid


    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this field."""
        return self.get_ctypes_struct()

    def get_ctypes_struct(self):
        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('data_chunks', POINTER(c_float)),
                        ('grid', POINTER(CGrid))]

        # Create and populate the c-struct object
        if not self.c_data.flags.c_contiguous:
            self.c_data = np.array(self.data.flatten(), dtype=self.data.dtype, order='C')
        self.c_data_ptr = self.c_data.ctypes.data_as(POINTER(c_float))

        cstruct = CField(self.grid.xdim, self.grid.ydim, self.grid.zdim,
                         self.grid.tdim, self.c_data_ptr,  # (POINTER(POINTER(c_float)) * len(self.c_data))(*self.c_data),
                         pointer(self.grid.ctypes_struct))
        return cstruct

    def __getitem__(self, key):
        return self.eval(*key)

    def eval(self, time, z, y, x):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        return 0

    def time_index(self, time):
        return 0

    def depth_index(self, depth, lat, lon):
        return 0

    def computeTimeChunk_simple(self, time_index):
        return self.computeTimeChunk(self.data, time_index)

    def computeTimeChunk(self, data, time_index):
        """
        (re-)computes the data field by updating it with a new time step
        :param data: input ndarray to be expanded / updated with new data
        :param time_index: time index the 'data' field need to be updated to
        :return: updated data np.ndarray
        """
        return self.data



class NestedField(Field):
    def __init__(self, fieldset=None):
        super(NestedField, self).__init__(fieldset)

class SummedField(Field):
    def __init__(self, fieldset=None):
        super(SummedField, self).__init__(fieldset)

class VectorField(Field):
    def __init__(self, fieldset=None):
        super(VectorField, self).__init__(fieldset)





class FieldSet(object):

    def __init__(self):
        self._data = []
        self.gridset = GridSet()
        # self.time = np.float64(0)

    def __del__(self):
        del self._data[:]
        self._data = None

    def get_fields(self):
        return self._data

    @property
    def fields(self):
        return self.get_fields()

    def append(self, obj):
        self._data.append(obj)

    @property
    def size(self):
        return len(self._data)

    def __len__(self):
        return len(self._data)

    def computeTimeChunk(self, time, dt):
        """Load a chunk of three data time steps into the FieldSet.
        This is used when FieldSet uses data imported from netcdf,
        with default option deferred_load. The loaded time steps are at or immediatly before time
        and the two time steps immediately following time if dt is positive (and inversely for negative dt)
        :param time: Time around which the FieldSet chunks are to be loaded. Time is provided as a double, relatively to Fieldset.time_origin
        :param dt: time step of the integration scheme
        """
        signdt = np.sign(dt)
        nextTime = np.infty if dt > 0 else -np.infty
        for f in self.fields:
            nextTime_loc = f.grid.computeTimeChunk(f, time, signdt)
            nextTime = min(nextTime, nextTime_loc) if signdt >= 0 else max(nextTime, nextTime_loc)

        if abs(nextTime) == np.infty or np.isnan(nextTime):  # Second happens when dt=0
            return nextTime
        else:
            nSteps = int((nextTime - time) / dt)
            if nSteps == 0:
                return nextTime
            else:
                return time + nSteps * dt

