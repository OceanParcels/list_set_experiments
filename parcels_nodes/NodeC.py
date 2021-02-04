from parcels_nodes.Node import Node
from parcels_nodes.wrapping import node
import ctypes
from parcels_nodes import package_globals
import numpy
import sys
from ctypes import POINTER, byref, pointer, c_void_p

# class NodeC(Node, node.NodeJIT):
class NodeC(Node):
    n_ptr = None
    init_node_c = None
    set_prev_ptr_c = None
    set_next_ptr_c = None
    # set_data_ptr_c = None
    reset_prev_ptr_c = None
    reset_next_ptr_c = None
    reset_data_ptr_c = None


    def __init__(self, prev=None, next=None, id=None, data=None):
        super().__init__(prev=prev, next=next, id=id, data=data)
        self.n_ptr = node.NodeJIT()
        self.init_node_c = node.init_node
        self.set_prev_ptr_c = node.set_prev_ptr
        self.set_next_ptr_c = node.set_next_ptr
        # self.set_data_ptr_c = node.set_data_ptr
        self.reset_prev_ptr_c = node.reset_prev_ptr
        self.reset_next_ptr_c = node.reset_next_ptr
        self.reset_data_ptr_c = node.reset_data_ptr

        # self.init_node_c(self)
        self.init_node_c(self.get_ptr())

        if self.prev is not None and isinstance(self.prev, NodeC):
            # self.set_prev_ptr_c(self, self.prev)
            self.set_prev_ptr_c(self.get_ptr(), self.prev.get_ptr())
        else:
            # self.reset_prev_ptr_c(self)
            self.reset_prev_ptr_c(self.get_ptr())
        if self.next is not None and isinstance(self.next, NodeC):
            # self.set_next_ptr_c(self, self.next)
            self.set_next_ptr_c(self.get_ptr(), self.next.get_ptr())
        else:
            # self.reset_next_ptr_c(self)
            self.reset_next_ptr_c(self.get_ptr())


        if self.data is not None:
            try:
                # self.set_data_ptr_c(self, self.data.cdata())
                node.set_data_ptr_numpy(self.get_ptr(), self.data.get_cptr())
            except (AttributeError, TypeError):
                # self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
                node.set_data_ptr_PyObject(self.get_ptr(), self.data)
        else:
            # self.reset_data_ptr_c(self)
            self.reset_data_ptr_c(self.get_ptr())

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        result = type(self)(prev=None, next=None, id=-1, data=None)
        result.id = self.id
        result.next = self.next
        result.prev = self.prev
        result.data = self.data

        result.init_node_c = self.init_node_c
        result.set_prev_ptr_c = self.set_prev_ptr_c
        result.set_next_ptr_c = self.set_next_ptr_c
        # result.set_data_ptr_c = self.set_data_ptr_c
        result.reset_prev_ptr_c = self.reset_prev_ptr_c
        result.reset_next_ptr_c = self.reset_next_ptr_c
        result.reset_data_ptr_c = self.reset_data_ptr_c
        # result.init_node_c(self)
        result.init_node_c(self.get_ptr())

        if result.prev is not None and isinstance(result.prev, NodeC):
            # result.set_prev_ptr_c(result, result.prev)
            result.set_prev_ptr_c(result.get_ptr(), result.prev.get_ptr())
        else:
            # result.reset_prev_ptr_c(result)
            result.reset_prev_ptr_c(result.get_ptr())
        if result.next is not None and isinstance(result.next, NodeC):
            # result.set_next_ptr_c(result, result.next)
            result.set_next_ptr_c(result.get_ptr(), result.next.get_ptr())
        else:
            # result.reset_next_ptr_c(result)
            result.reset_next_ptr_c(result.get_ptr())

        if result.data is not None:
            try:
                # self.set_data_ptr_c(self, self.data.cdata())
                node.set_data_ptr_numpy(result.get_ptr(), result.data.get_cptr())
            except (AttributeError, TypeError):
                # self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
                node.set_data_ptr_PyObject(result.get_ptr(), result.data)
        else:
            # result.reset_data_ptr_c(result)
            result.reset_data_ptr_c(result.get_ptr())
        return result

    def __del__(self):
        # print("NodeJIT.del() [id={}] is called.".format(self.id))
        self.unlink()
        del self.data
        package_globals.idgen.releaseID(self.id)
        del self.n_ptr
        self.n_ptr = None

    def get_ptr(self):
        return self.n_ptr

    def unlink(self):
        if self.prev is not None:
            # prev is STH
            if self.next is not None:
                # next is STH
                self.prev.set_next(self.next)
                self.next.set_prev(self.prev)
            else:
                # next is NULL
                self.reset_next_ptr_c(self.prev.get_ptr())
        else:
            # prev is NULL
            if self.next is not None:
                # next is STH
                self.reset_prev_ptr_c(self.next.get_ptr())
            else:
                # next is NULL
                pass

        # print("NodeJIT.unlink() [id={}] is called.".format(self.id))
        # if self.prev is not None:
        #     if self.next is not None:
        #         # self.prev.set_next_c(self.next)
        #         self.prev.set_next_c(self.next.get_ptr())
        #     else:
        #         # self.reset_next_ptr_c(self.prev)
        #         self.reset_next_ptr_c(self.prev.get_ptr())
        # if self.next is not None:
        #     if self.prev is not None:
        #         # self.next.set_prev_c(self.prev)
        #         self.next.set_prev_c(self.prev.get_ptr())
        #     else:
        #         # self.reset_prev_ptr_c(self.next)
        #         self.reset_prev_ptr_c(self.next.get_ptr())

        if self is not None:
            # self.reset_prev_ptr_c(self)
            self.reset_prev_ptr_c(self.get_ptr())
            # self.reset_next_ptr_c(self)
            self.reset_next_ptr_c(self.get_ptr())
            # self.reset_data_ptr_c(self)
            self.reset_data_ptr_c(self.get_ptr())

        self.prev = None
        self.next = None

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

    def __sizeof__(self):
        return super().__sizeof__() # +sys.getsizeof(self._fields_)

    def set_data(self, data):
        super().set_data(data)
        self.update_data()

    def set_prev(self, prev):
        super().set_prev(prev)
        self.update_prev()

    def set_next(self, next):
        super().set_next(next)
        self.update_next()

    def update_prev(self):
        if self.prev is not None and isinstance(self.prev, NodeC):
            # self.set_prev_ptr_c(self, self.prev)
            self.set_prev_ptr_c(self.get_ptr(), self.prev.get_ptr())
        else:
            # self.reset_prev_ptr_c(self)
            self.reset_prev_ptr_c(self.get_ptr())

    def update_next(self):
        if self.next is not None and isinstance(self.next, NodeC):
            # self.set_next_ptr_c(self, self.next)
            self.set_next_ptr_c(self.get_ptr(), self.next.get_ptr())
        else:
            # self.reset_next_ptr_c(self)
            self.reset_next_ptr_c(self.get_ptr())

    def update_data(self):
        if self.data is not None:
            try:
                # self.set_data_ptr_c(self, self.data.cdata())
                node.set_data_ptr_numpy(self.get_ptr(), self.data.get_cptr())
            except (AttributeError, TypeError):
                # self.set_data_ptr_c(self, ctypes.cast(self.data, ctypes.c_void_p))
                node.set_data_ptr_PyObject(self.get_ptr(), self.data)
        else:
            # self.reset_data_ptr_c(self)
            self.reset_data_ptr_c(self.get_ptr())