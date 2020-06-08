from LinkedList import *
from Node import *
import package_globals
from time import time as real_time
from time import perf_counter
from time import process_time
# import random
from numpy import random
import numpy
from copy import deepcopy

from particle import JITParticle, ScipyParticle
from parcels_mocks import *

from particleset_node import ParticleSet


class divide_n_conquer_list:
    _nodes = []

    def __init__(self):
        self._nodes = []

    def add(self, number):
        self._nodes.append(number)

    def set_nodes(self, mlist):
        self._nodes = mlist

    def get_by_id(self, id):
        lower = 0
        upper = len(self._nodes)-1
        pos = lower + int((upper-lower)/2.0)
        current_node = self._nodes[pos]
        while current_node != id:
            print("lower: {}, upper: {}, pos {}".format(lower, upper, pos))
            if id < current_node:
                lower = lower
                upper = pos-1
                pos = lower + int((upper-lower)/2.0)
            else:
                lower = pos
                upper = upper
                pos = lower + int((upper-lower)/2.0)+1
            current_node = self._nodes[pos]
        return current_node



if __name__ == '__main__':
    print("====== Test Divide-n-Conquer search ======")
    mlist = divide_n_conquer_list()
    vlist = [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711]
    mlist.set_nodes(vlist)
    for i in range(0, 10):
        rpos = random.randint(low=0, high=len(vlist))
        print("---- test finding '{}' ... ----".format((rpos, vlist[rpos])))
        mlist.get_by_id(vlist[rpos])


    print("====== Test Spatial ID generator ======")
    package_globals.spat_idgen.setDepthLimits(0.0, 75)
    package_globals.spat_idgen.setTimeLine(0.0, 365.0)
    id1 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
    id2 = package_globals.spat_idgen.getID(0.0, 0.0, 20.0, 0.0)
    id3 = package_globals.spat_idgen.getID(4.895168, 52.370216, 12.0, 0.0)  # Amsterdam
    id4 = package_globals.spat_idgen.getID(-43.172897, -22.906847, 12.0, 0.0)   # Rio de Janeiro
    print("Test-ID 1:         {}".format(numpy.binary_repr(id1, width=64)))
    print("Test-ID 2:         {}".format(numpy.binary_repr(id2, width=64)))
    print("Test-ID Amsterdam: {}".format(numpy.binary_repr(id3, width=64)))
    print("Test-ID Rio:       {}".format(numpy.binary_repr(id4, width=64)))
    print("===========================================================================")

    random.seed(int(real_time().is_integer()))
    with_numpy = True
    #N = 2 ** 20
    #N = 2 ** 18
    N = 2 ** 16
    #N = 2 ** 14
    #N = 2 ** 12
    #N = 2 ** 10
    #N = 2 ** 4

    fieldset = FieldSet()
    fieldset.append(Field())
    fieldset.append(Field())
    fieldset.gridset.append(Grid())


    package_globals.idgen.preGenerateIDs(N)
    package_globals.idgen.permuteIDs()

    stime = process_time()

    # dbl_list = OrderedList(dtype=NodeJIT)
    # dbl_list = OrderedList(dtype=Node)
    real_list = RealList(dtype=NodeJIT)
    # real_list = RealList(dtype=Node)
    print("Real list created.")

    while len(real_list) < N:
        n = len(real_list)
        index = package_globals.idgen.nextID()
        node = NodeJIT(id=int(index), data=JITParticle(lon=random.random_sample(), lat=random.random_sample(), pid=int(index), fieldset=fieldset, depth=random.random_sample(), time=0))
        # node = Node(id=int(index))

        real_list.add(node)
        # real_list.insert(node)
    etime = process_time()
    print("Time adding {} particles (RealList): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(dbl_list)))
    #print([str(v) for v in dbl_list])

    stime = process_time()
    iter = 0
    while iter < N:
        #print("=========================")
        n = len(real_list)
        index = numpy.random.randint(0, n)
        # search_node = real_list[index]
        # real_list.remove(search_node)

        search_node = real_list.pop(index)
        #search_node = real_list.popitem(index)


        # del real_list[index]
        # index = package_globals.idgen.nextID()
        # search_node = NodeJIT(id=index)
        # search_node = Node(id=index)

        real_list.add(search_node)
        # real_list.insert(search_node)

        iter += 1
    etime = process_time()
    print("Time delete and insert {} particles (RealList): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(dbl_list)))
    #print([str(v) for v in dbl_list])
    del real_list
    real_list = None
    print("===========================================================================")

    stime = process_time()
    ord_list = OrderedList(dtype=NodeJIT)
    # ord_list = OrderedList(dtype=Node)
    print("Ordered list created.")
    while len(ord_list) < N:
        n = len(ord_list)
        index = package_globals.idgen.nextID()
        node = NodeJIT(id=int(index), data=JITParticle(lon=random.random_sample(), lat=random.random_sample(), pid=int(index), fieldset=fieldset, depth=random.random_sample(), time=0))
        # node = Node(id=int(index))

        # ord_list.add(node)
        ord_list.insert(node)
    etime = process_time()
    print("Time adding {} particles (OrderedList): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(ord_list)))
    #print([str(v) for v in ord_list])

    stime = process_time()
    iter = 0
    while iter < N:
        #print("=========================")
        n = len(ord_list)
        index = numpy.random.randint(0, n)
        #search_node = ord_list[index]
        #ord_list.remove(search_node)

        # search_node = ord_list.pop(index)
        search_node = ord_list.popitem(index)


        # del ord_list[index]
        # index = package_globals.idgen.nextID()
        # search_node = NodeJIT(id=index)
        # search_node = Node(id=index)

        # ord_list.add(search_node)
        ord_list.insert(search_node)

        iter += 1
    # print([str(v) for v in dbl_list])
    etime = process_time()
    print("Time delete and insert {} particles (OrderedList): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(ord_list)))
    del ord_list
    dbl_list = None
    print("===========================================================================")

    if with_numpy:  # do that with , data=JITParticle(lon=random.random_sample(), lat=random.random_sample(), depth=random.random_sample(), time=0)  --  SOMEHOW
        # -- coords = numpy.random.random((1,3)).astype(dtype=numpy.float64, order='C')
        # -- np_list = numpy.array(coords, dtype=numpy.float64, order='C')
        # == coords = numpy.zeros(1, dtype=JITParticle, order='C')
        # == coords[0] = JITParticle(lon=random.random_sample(), lat=random.random_sample(), pid=int(index), fieldset=fieldset, depth=random.random_sample(), time=0)
        # == np_list = numpy.array(coords, dtype=JITParticle, order='C')

        np_list = numpy.array([], dtype=JITParticle, order='C')
        # np_list = numpy.array([], dtype=JITParticle)
        # np_list = numpy.zeros(1, dtype=JITParticle, order='C')
        stime = process_time()
        print("NumPy nD-Array created.")
        while np_list.shape[0] < N:
            index = package_globals.idgen.nextID()
            # coords = numpy.random.random((1, 3)).astype(dtype=numpy.float64, order='C')
            coords = numpy.zeros(1, dtype=JITParticle, order='C')
            # coords = numpy.zeros(1, dtype=JITParticle)
            coords[0] = JITParticle(lon=random.random_sample(), lat=random.random_sample(), pid=int(index), fieldset=fieldset, depth=random.random_sample(), time=0.)
            np_list = numpy.concatenate((np_list, coords))
            # if np_list.shape[0] < 10:
            #     print(np_list[-1])
        etime = process_time()
        print("Time adding {} particles (NumPy Array): {}".format(N, etime-stime))

        stime = process_time()
        iter = 0
        while iter < N:
            n = np_list.shape[0]
            index = numpy.random.randint(0, n)
            # coords = numpy.reshape(np_list[index,:],(1,3), order='C')
            coords = numpy.zeros(1, dtype=JITParticle, order='C')
            # coords = numpy.zeros(1, dtype=JITParticle)
            coords[0] = np_list[index]
            np_list = numpy.delete(np_list,index,0)
            np_list = numpy.concatenate((np_list, coords))
            iter += 1
        etime = process_time()
        print("Time delete and insert {} particles (NumPy Array): {}".format(N, etime-stime))
        del np_list
    print("===========================================================================")

    # nclass = Node
    nclass = NodeJIT
    # pclass = ScipyParticle
    pclass = JITParticle
    real_pset = ParticleSet(fieldset, pclass, lonlatdepth_dtype=numpy.float32)
    print("Real ParticleSet created.")

    while len(real_pset) < N:
        n = len(real_pset)
        index = package_globals.idgen.nextID()
        node = NodeJIT(id=int(index), data=JITParticle(lon=random.random_sample(), lat=random.random_sample(), pid=int(index), fieldset=fieldset, depth=random.random_sample(), time=0))
        # node = Node(id=int(index))
        real_pset.add(node)
    etime = process_time()
    print("Time adding {} particles (Real ParticleSet): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(dbl_list)))
    #print([str(v) for v in dbl_list])

    stime = process_time()
    iter = 0
    while iter < N:
        n = len(real_pset)
        index = numpy.random.randint(0, n)
        search_node = real_pset.pop(index)
        real_pset.add(search_node)

        iter += 1
    etime = process_time()
    print("Time delete and insert {} particles (Real ParticleSet): {}".format(N, etime-stime))
    #print("len(list): {}".format(len(dbl_list)))
    #print([str(v) for v in dbl_list])
    del real_pset
    real_pset = None

