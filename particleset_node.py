from LinkedList import *
from particle import ScipyParticle, JITParticle
import numpy as np


class ParticleSet(object):
    _nodes = None
    _pclass = ScipyParticle
    _nclass = Node
    _ptype = None

    # def __init__(self, fieldset, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
    def __init__(self, fieldset, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        if lonlatdepth_dtype is not None:
            self.lonlatdepth_dtype = lonlatdepth_dtype
        else:
            self.lonlatdepth_dtype = np.float32
        JITParticle.set_lonlatdepth_dtype(self.lonlatdepth_dtype)

        self._pclass = pclass
        self._ptype = self._pclass.getPType()
        if self._ptype.uses_jit:
            self._nclass = NodeJIT
        else:
            self._nclass = Node
        self._nodes = RealList(dtype=self._nclass)

        # fill / initialize the list

    def cptr(self, index):
        if self._ptype.uses_jit:
            node = self._nodes[index]
            return node.data.get_cptr()
        else:
            return None

    @property
    def size(self):
        return len(self._nodes)

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
        return self.get(index).get_data()

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
        if isinstance(ndata, list) or (isinstance(ndata, np.ndarray) and ndata.dtype is np.int32):
            pass # remove multiple instances
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
        for ndata in ndata_array:
            self.remove_entity(ndata)

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

    def execute(self):
        pass










