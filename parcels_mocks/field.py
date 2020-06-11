from .grid import *
import numpy as np


class Field(object):
    data = None

    def __init__(self, fieldset=None):
        self.fieldset = fieldset
        self.data = np.empty((2,2), dtype=np.float32)

    def __getitem__(self, key):
        return self.eval(*key)

    def eval(self, time, z, y, x):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        return 0

    def computeTimeChunk_simple(self, time_index):
        return self.computeTimeChunk()

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

    def __del__(self):
        del self._data[:]
        self._data = None

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
        pass

