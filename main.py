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

if __name__ == '__main__':
    random.seed(int(real_time().is_integer()))
    with_numpy = True
    #N = 2 ** 20
    N = 2 ** 18
    #N = 2 ** 16
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

