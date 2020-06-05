

class Field(object):

    def __init__(self, fieldset=None):
        self.fieldset = fieldset

    def __getitem__(self, key):
        return self.eval(*key)

    def eval(self, time, z, y, x):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        return 0


class Grid(object):

    def __init__(self):
        pass



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

    def computeTimeChunk(self, time_value, time_index):
        pass


class GridSet(object):

    def __init__(self):
        self._data = []

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


