import time as time_module
from datetime import date
from datetime import datetime as dtime
from datetime import timedelta as delta

from LinkedList import *
import package_globals

from particle import ScipyParticle, JITParticle
from parcels_mocks import Grid, Field, GridSet, FieldSet
import numpy as np

from kernelbase import BaseNoFieldKernel, BaseFieldKernel
from kernel import DoNothing, NodeNoFieldKernel, NodeFieldKernel
from kernel import NodeNoFieldKernel as Kernel
from parcels_mocks.status import StatusCode as ErrorCode


class RepeatParameters(object):
    _n_pts = 0
    _lon = []
    _lat = []
    _depth = []
    _maxID = None
    _pclass = ScipyParticle
    _partitions = None
    kwargs = None

    def __init__(self, pclass=JITParticle, lon=[], lat=[], depth=[], partitions=None, pid_orig=None, **kwargs):
        self._lon = lon
        self._lat = lat
        self._maxID = pid_orig # pid - pclass.lastID
        self._depth = depth
        assert type(self._lon)==type(self._lat)==type(self._depth)
        if isinstance(self._lon, list):
            self._n_pts = len(self._lon)
        elif isinstance(self._lon, np.ndarray):
            self._n_pts = self._lon.shape[0]
        self._pclass = pclass
        self._partitions = partitions
        self.kwargs = kwargs

    def get_num_pts(self):
        return self._n_pts

    def get_lon(self):
        return self._lon

    def get_longitude(self, index):
        return self._lon[index]

    def get_lat(self):
        return self._lat

    def get_latitude(self, index):
        return self._lat[index]

    def get_depth(self):
        return self._depth

    def get_depth_value(self, index):
        return self._depth[index]

    def get_pid(self):
        return self._maxID

    def get_particle_id(self, index):
        if self._maxID is None:
            return None
        return self._maxID+index

    def get_pclass(self):
        return

    def get_partitions(self):
        return self._partitions



class ParticleSet(object):
    _nodes = None
    _pclass = ScipyParticle
    _nclass = Node
    _kclass = NodeFieldKernel
    _ptype = None
    _fieldset = None
    _kernel = None
    lonlatdepth_dtype = None

    def __init__(self, fieldset = FieldSet(), pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        self._fieldset = fieldset
        if lonlatdepth_dtype is not None:
            self.lonlatdepth_dtype = lonlatdepth_dtype
        else:
            self.lonlatdepth_dtype = np.float32
        JITParticle.set_lonlatdepth_dtype(self.lonlatdepth_dtype)
        pid = None if pid_orig is None else pid_orig + pclass.lastID

        self._pclass = pclass
        self._kclass = NodeNoFieldKernel
        self._kernel = None
        self._ptype = self._pclass.getPType()
        if self._ptype.uses_jit:
            self._nclass = NodeJIT
        else:
            self._nclass = Node
        self._nodes = RealList(dtype=self._nclass)

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        rdata_available = True
        rdata_available &= (lon is not None) and (isinstance(lon, list) or isinstance(lon, np.ndarray))
        rdata_available &= (lat is not None) and (isinstance(lat, list) or isinstance(lat, np.ndarray))
        rdata_available &= (depth is not None) and (isinstance(depth, list) or isinstance(depth, np.ndarray))
        rdata_available &= (time is not None) and (isinstance(time, list) or isinstance(time, np.ndarray))
        if self.repeatdt and rdata_available:
            self.repeat_starttime = self._fieldset.gridset.dimrange('full_time')[0] if time is None else time[0]
            self.rparam = RepeatParameters(self._pclass, lon, lat, depth, None, None if pid is None else (pid - pclass.lastID), **kwargs )

        # fill / initialize the list

    def cptr(self, index):
        if self._ptype.uses_jit:
            node = self._nodes[index]
            return node.data.get_cptr()
        else:
            return None

    def empty(self):
        return self.size > 0

    def begin(self):
        """
        Returns the begin of the linked particle list (like C++ STL begin() function)
        :return: begin Node (Node whose prev element is None); returns None if ParticleSet is empty
        """
        if not self.empty():
            node = self._nodes[0]
            while node.prev is not None:
                node = node.prev
            return node
        return None

    def end(self):
        """
        Returns the end of the linked partile list. UNLIKE in C++ STL, it returns the last element (valid element),
        not the element past the last element (invalid element). (see http://www.cplusplus.com/reference/list/list/end/)
        :return: end Node (Node whose next element is None); returns None if ParticleSet is empty
        """
        if not self.empty():
            node = self._nodes[self.size-1]
            while node.next is not None:
                node = node.next
            return node
        return None

    def set_kernel_class(self, kclass):
        self._kclass = kclass

    @property
    def size(self):
        return len(self._nodes)

    @property
    def data(self):
        return self._nodes

    def __len__(self):
        return self.size

    def __repr__(self):
        result = "\n"
        node = self._nodes[0]
        while node.prev is not None:
            node = node.prev
        while node.next is not None:
            result += str(node) + "\n"
            node = node.next
        result += str(node) + "\n"
        return result
        #return "\n".join([str(p) for p in self])

    def get(self, index):
        return self.get_by_index(index)

    def get_by_index(self, index):
        return self.__getitem__(index)

    def get_by_id(self, id):
        """
        divide-and-conquer search of SORTED list
        :param id: search Node ID
        :return: Node attached to ID
        """
        lower = 0
        upper = len(self._nodes)-1
        pos = lower + int((upper-lower)/2.0)
        current_node = self._nodes[pos]
        while current_node.id != id:
            if id < current_node.id:
                lower = lower
                upper = pos-1
                pos = lower + int((upper-lower)/2.0)
            else:
                lower = pos
                upper = upper
                pos = lower + int((upper-lower)/2.0)+1
            current_node = self._nodes[pos]
        return current_node

    def get_particle(self, index):
        return self.get(index).data

    def retrieve_item(self, key):
        return self.get(key)

    def __getitem__(self, key):
        if key >= 0 and key < len(self._nodes):
            return self._nodes[key]
        return None

    def __setitem__(self, key, value):
        """

        :param key: index (int; np.int32) or Node
        :param value: particle data of the particle class
        :return: modified Node
        """
        try:
            assert (isinstance(value, self._pclass))
        except AssertionError:
            print("setting value not of type '{}'".format(str(self._pclass)))
            exit()
        if isinstance(key, int) or isinstance(key, np.int32):
            search_node = self._nodes[key]
            search_node.set_data(value)
        elif isinstance(key, self._nclass):
            assert (key in self._nodes)
            key.set_data(value)

    def __iadd__(self, pdata):
        self.add(pdata)
        return self

    def add(self, pdata):
        """
        Adds the new data in the list - position is auto-determined (because of sorted-list nature)
        :param pdata: new Node or pdata
        :return: index of inserted node
        """
        index = -1
        if isinstance(pdata, self._nclass):
            self._nodes.add(pdata)
            index = self._nodes.bisect_right(pdata)
        else:
            index = package_globals.idgen.nextID()
            pdata.id = int(index)
            node = NodeJIT(id=int(index), data=pdata)
            self._nodes.add(node)
            index = self._nodes.bisect_right(node)
        if index > 0:
            # return self._nodes[index]
            return index
        return None

    def remove(self, ndata):
        if ndata is None:
            pass
        # if isinstance(ndata, list) or (isinstance(ndata, np.ndarray) and ndata.dtype is np.int32):
        elif isinstance(ndata, list) or isinstance(ndata, np.ndarray):
            self.remove_entities(ndata) # remove multiple instances
        else:
            self.remove_entity(ndata)

    def remove_entity(self, ndata):
        if isinstance(ndata, int) or isinstance(ndata, np.int32):
            del self._nodes[ndata]
            # search_node = self._nodes[ndata]
            # self._nodes.remove(search_node)
        elif isinstance(ndata, self._nclass):
            self._nodes.remove(ndata)
        elif isinstance(ndata, self._pclass):
            node = self.get_by_id(ndata.id)
            self._nodes.remove(node)

    def remove_entities(self, ndata_array):
        rm_list = ndata_array
        if len(ndata_array) <= 0:
            return
        if isinstance(rm_list[0], int) or isinstance(rm_list[0], np.int32) or isinstance(rm_list[0], np.int64):
            rm_list = []
            for index in ndata_array:
                rm_list.append(self.get_by_index(index))
        for ndata in rm_list:
            self.remove_entity(ndata)

    def get_deleted_item_indices(self):
        indices = [i for i, n in enumerate(self._nodes) if n.data.state is ErrorCode.Delete]
        return indices

    def remove_deleted_items(self):
        node = self.begin()
        while node is not None:
            next_node = node.next
            if node.data.state is ErrorCode.Delete:
                self._nodes.remove(node)
            node = next_node

    def pop(self, idx=-1, deepcopy_elem=False):
        return self._nodes.pop(idx, deepcopy_elem)

    def insert(self, node_or_pdata):
        """
        Inserts new data in the list - position is auto-determined
        :param node_or_pdata: new Node or pdata
        :return: index of inserted node
        """
        return self.add(node_or_pdata)

    def merge(self, key1, key2):
        pass

    def split(self, key):
        """
        splits a node, returning the result 2 new nodes
        :param key: index (int; np.int32), Node
        :return: 'node1, node2' or 'index1, index2'
        """
        pass

    def execute(self, pyfunc=DoNothing, endtime=None, runtime=None, dt=1.,
                recovery=None, output_file=None, verbose_progress=None):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double. [DURATION]
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        """

        # check if pyfunc has changed since last compile. If so, recompile
        if self._kernel is None or (self._kernel.pyfunc is not pyfunc and self._kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, self._kclass):
                self._kernel = pyfunc
            else:
                self._kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self._ptype.uses_jit:
                self._kernel.remove_lib()
                cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
                # self._kernel.compile(compiler=GNUCompiler(cppargs=cppargs))
                self._kernel.compile(compiler=GNUCompiler(cppargs=cppargs, incdirs=[os.path.join(package_globals.get_package_dir(), 'include'), "."], libdirs=[".", ], libs=["node"]))
                self._kernel.load_lib()

        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, dtime):
            endtime = np.datetime64(endtime)
        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        outputdt = output_file.outputdt if output_file else np.infty
        if isinstance(outputdt, delta):
            outputdt = outputdt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'

        # Set particle.time defaults based on sign of dt, if not set at ParticleSet construction
        piter = 0
        while piter < len(self._nodes):
            pdata = self._nodes[piter].data
            if np.isnan(pdata.time):
                mintime, maxtime = self._fieldset.gridset.dimrange('time_full')
                pdata.time = mintime if dt >= 0 else maxtime
                self._nodes[piter].set_data(pdata)

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        # ====================================== #
        # ==== EXPENSIVE LIST COMPREHENSION ==== #
        # ====================================== #
        _starttime = min([p.time for p in self]) if dt >= 0 else max([p.time for p in self])
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self._fieldset.gridset.dimrange('time_full')
            endtime = maxtime if dt >= 0 else mintime

        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime

        # Initialise particle timestepping
        #for p in self:
        #    p.dt = dt
        piter = 0
        while piter < len(self._nodes):
            pdata = self._nodes[piter].data
            pdata.dt = dt
            self._nodes[piter].set_data(pdata)

        # First write output_file, because particles could have been added
        if output_file is not None:
            output_file.write(self, _starttime)

        time = _starttime
        if self.repeatdt and self.rparam is not None:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt
        next_input = self._fieldset.computeTimeChunk(time, np.sign(dt))

        tol = 1e-12
        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            if dt > 0:
                time = min(next_prelease, next_input, next_output, endtime)
            else:
                time = max(next_prelease, next_input, next_output, endtime)
            self._kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file)
            if abs(time-next_prelease) < tol:
                add_iter = 0
                while add_iter < self.rparam.get_num_pts():
                    gen_id = self.rparam.get_particle_id(add_iter)
                    pindex = package_globals.idgen.nextID() if gen_id is None else gen_id
                    pdata = JITParticle(lon=self.rparam.get_longitude(add_iter), lat=self.rparam.get_latitude(add_iter), pid=pindex, fieldset=self._fieldset, depth=self.rparam.get_depth_value(add_iter), time=time[add_iter])
                    pdata.dt = dt
                    self.add(self._nclass(id=pindex, data=pdata))
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol:
                if output_file is not None:
                    output_file.write(self, time)
                next_output += outputdt * np.sign(dt)
            if time != endtime:
                next_input = self._fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break

        if output_file is not None:
            output_file.write(self, time)



    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet
        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return self._kclass(self._fieldset, self._ptype, pyfunc=pyfunc, c_include=c_include, delete_cfiles=delete_cfiles)

    #def ParticleFile(self, *args, **kwargs):
    #    """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
    #    object from the ParticleSet"""
    #    return ParticleFile(*args, particleset=self, **kwargs)








