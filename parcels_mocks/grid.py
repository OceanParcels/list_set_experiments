import numpy as np


class Grid(object):
    _cstruct = None

    def __init__(self):
        pass

    @property
    def cstruct(self):
        return self._cstruct




class GridSet(object):
    time = []

    def __init__(self):
        self._data = []
        self.time = []

    def __del__(self):
        del self._data[:]
        self._data = None

    def append(self, obj):
        self._data.append(obj)

    def set_time(self, min_time, max_time, grid_dt):
        self.time = np.arange(min_time, max_time, grid_dt, dtype=np.float64)

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
