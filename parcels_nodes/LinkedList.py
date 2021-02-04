from parcels_nodes.Node import *
from parcels_nodes import package_globals
from sortedcontainers import SortedList
from copy import copy, deepcopy
import gc


# ========================== #
# = Verdict: nice try, but = #
# = overrides to the del() = #
# = function won't work.   = #
# ========================== #
class RealList(SortedList):
    dtype = None

    def __init__(self, iterable=None, dtype=Node):
        super(RealList, self).__init__(iterable)
        self.dtype=dtype

    def __del__(self):
        self.clear()

    def clear(self):
        """Remove all the elements from the list."""
        n = self.__len__()
        # print("# remaining items: {}".format(n))
        if n > 0:
            print("Deleting {} elements ...".format(n))
            while (n > 0):
                # val = self.pop(); del val
                # super().__delitem__(n-1)
                # self.pop()

                self.__getitem__(-1).unlink()
                self.__delitem__(-1)
                n = self.__len__()
                # print("Deleting {} elements ...".format(n))
        # gc.collect()
        super()._clear()

    def __new__(cls, iterable=None, key=None, load=1000, dtype=Node):
        return object.__new__(cls)

    def add(self, val):
        assert type(val)==self.dtype
        if isinstance(val, Node):
            n = self.__len__()
            index = self.bisect_right(val)
            if index < n:
                next_node = self.__getitem__(index)
            else:
                next_node = None
            if index > 0:
                prev_node = self.__getitem__(index-1)
            else:
                prev_node = None
            if next_node is not None:
                next_node.set_prev(val)
                val.set_next(next_node)
            if prev_node is not None:
                prev_node.set_next(val)
                val.set_prev(prev_node)
            super().add(val)
        elif isinstance(val, int):
            n = self.__len__()
            index = self.bisect_right(val)
            if index < n:
                next_node = self.__getitem__(index)
            else:
                next_node = None
            if index > 0:
                prev_node = self.__getitem__(index-1)
            else:
                prev_node = None
            new_node = self.dtype(prev=prev_node, next=next_node, id=val)
            super().add(new_node)
        else:
            new_node = self.dtype(data=val)
            super().add(new_node)
            n = self.__len__()
            index = self.index(new_node)
            if index < (n-2):
                next_node = self.__getitem__(index+1)
            else:
                next_node = None
            if index > 0:
                prev_node = self.__getitem__(index-1)
            else:
                prev_node = None
            if next_node is not None:
                next_node.set_prev(new_node)
                new_node.set_next(next_node)
            if prev_node is not None:
                prev_node.set_next(new_node)
                new_node.set_prev(prev_node)

    def append(self, val):
        self.add(val)

    def pop(self, idx=-1, deepcopy_elem=False):
        """
        Because we expect the return node to be of use,
        the actual node is NOT physically deleted (by calling
        the destructor). pop() only dereferences the object
        in the list. The parameter 'deepcopy_elem' can be set so as
        to physically delete the list object and return a deep copy (unlisted).
        :param idx:
        :param deepcopy_elem:
        :return:
        """
        if deepcopy_elem:
            result = deepcopy(self.__getitem__(idx))
            val = super().pop(idx)
            del val
            return result
        return super().pop(idx)






class OrderedList(object):
    _list = None
    dtype = None

    def __init__(self, iterable=None, dtype=Node):
        self.dtype=dtype
        if isinstance(iterable, dict):
            iterable = None
        if iterable is not None:
            try:
                assert len(iterable) > 0
                assert isinstance(iterable[0], self.dtype)
            except AssertionError:
                iterable = None
        self._list = SortedList(iterable)

    def __del__(self):
        n = len(self._list)
        # print("# remaining items: {}".format(n))
        if n > 0:
            print("Deleting {} elements ...".format(n))
            while (n > 0):
                # val = self.pop(); del val
                # super().__delitem__(n-1)
                # self.pop()
                item = self._list[-1]
                item.unlink()
                del self._list[-1]
                n = len(self._list)
                # print("Deleting {} elements ...".format(n))
        #gc.collect()
        del self._list

    def __add__(self, other):
        """
        Does exclusively in-place addition
        :param other: new insertion object (e.g. Node)
        :return: this object
        """
        assert type(other)==self.dtype
        if isinstance(other, Node):
            n = len(self._list)
            index = self._list.bisect_right(other)  # .bisect_right(val) || .index(other)
            if index < n:
                next_node = self._list[index]
            else:
                next_node = None
            if index > 0:
                prev_node = self._list[index-1]
            else:
                prev_node = None
            if next_node is not None:
                next_node.set_prev(other)
                other.set_next(next_node)
            if prev_node is not None:
                prev_node.set_next(other)
                other.set_prev(prev_node)
            self._list.add(other)
        elif isinstance(other, int):
            n = len(self._list)
            index = self._list.bisect_right(other)
            if index < n:
                next_node = self._list[index]
            else:
                next_node = None
            if index > 0:
                prev_node = self._list[index-1]
            else:
                prev_node = None
            new_node = self.dtype(prev=prev_node, next=next_node, id=other)
            self._list.add(new_node)
        else:
            new_node = self.dtype(data=other)
            self._list.add(new_node)
            n = len(self._list)
            index = self._list.index(new_node)
            if index < (n-2):
                next_node = self._list[index+1]
            else:
                next_node = None
            if index > 0:
                prev_node = self._list[index-1]
            else:
                prev_node = None
            if next_node is not None:
                next_node.set_prev(new_node)
                new_node.set_next(next_node)
            if prev_node is not None:
                prev_node.set_next(new_node)
                new_node.set_prev(prev_node)
        return self

    def __iadd__(self, other):
        """
        In-place addition (as syntactical)
        :param other: new insertion object (e.g. Node)
        :return: this object
        """
        return self.__add__(other)

    def __len__(self):
        return len(self._list)

    def __delitem__(self, order_element):
        if isinstance(order_element, int):
            index = order_element
        else:
            try:
                index = self._list.index(order_element)
            except ValueError:
                err_msg = "Requested object {} not part of OrderedList.".format(other)
                raise IndexError(err_msg)
        del self._list[index]


    def __index__(self, other):
        try:
            index = self._list.index(other)
        except ValueError:
            err_msg = "Requested object {} not part of OrderedList.".format(other)
            raise IndexError(err_msg)
        return index

    def pop(self):
        self.pop_back()

    def popitem(self, index):
        n = len(self._list)
        if n > 0:
            return self._popitem_internal_(index)
        return None

    def pop_front(self):
        n = len(self._list)
        if n > 0:
            return self._popitem_internal_(0)
        return None

    def pop_back(self):
        n = len(self._list)
        if n > 0:
            return self._popitem_internal_()
        return None

    def _popitem_internal_(self, idx=-1, deepcopy_elem=False):
        """
        Because we expect the return node to be of use,
        the actual node is NOT physically deleted (by calling
        the destructor). pop() only dereferences the object
        in the list. The parameter 'deepcopy_elem' can be set so as
        to physically delete the list object and return a deep copy (unlisted).
        :param idx:
        :param deepcopy_elem:
        :return:
        """
        if deepcopy_elem:
            result = deepcopy(self._list[idx])
            val = self._list.pop(idx)
            del val
            return result
        return self._list.pop(idx)

    def insert(self, other):
        self.__add__(other)

    def remove(self, other):
        self.__delitem__(other)

    def __getitem__(self, order_element):
        if isinstance(order_element, int):
            index = order_element
        else:
            try:
                index = self._list.index(order_element)
            except ValueError:
                err_msg = "Requested object {} not part of OrderedList.".format(other)
                raise IndexError(err_msg)
        return self._list[index]

    def get(self, order_element):
        return self.__getitem__(order_element)

    def clear(self):
        n = len(self._list)
        # print("# remaining items: {}".format(n))
        if n > 0:
            # print("Deleting {} elements ...".format(n))
            while (n > 0):
                self._list[-1].unlink()
                del self._list[-1]
                n = len(self._list)
                # print("Deleting {} elements ...".format(n))




class DoubleLinkedList(object):
    tail = None
    head = None
    nodeType = type(Node)
    n_elem = 0

    def __init__(self, nodeType=None):
        self.n_elem = 0
        if nodeType is None:
            self.nodeType = type(Node)
        else:
            self.nodeType = type(nodeType)

    def __init__(self, node):
        self.n_elem = 0
        assert(node is not None)
        assert(isinstance(node, Node))
        self.tail = node
        p_node = None
        c_node = self.tail
        while c_node is not None:
            p_node = c_node
            c_node = c_node.next
        self.head = p_node
        self.n_elem += 1
        self.nodeType = type(node)

    def __len__(self):
        return self.n_elem

    def traverse(self):
        pass

    def get(self, value):
        if isinstance(value, int):
            return self.get_by_id(value)
        else:
            return self.get_by_data(value)

    def get_by_data(self, node_data):
        c_node = self.tail
        while c_node != None and c_node.data != node_data:
            c_node = c_node.next
        return c_node

    def get_by_id(self, node_id):
        c_node = self.tail
        while c_node != None and c_node.id != node_id:
            c_node = c_node.next
        return c_node

    def insert(self, value):
        if isinstance(value, Node):
            self.insert_node(value)
        elif isinstance(value, int):
            node = self.nodeType(id=package_globals.idgen.nextID(), data=value)
            self.insert_node(node)
        else:
            TypeError("'value' not accepted type (type='{}')".format(type(value)))

    def insert_node(self, node):
        skip = False
        if self.tail is None:
            self.tail = node
            skip = True
        if self.head is None:
            self.head = node
            skip = True
        if skip:
            return
        c_node = self.tail
        while c_node != None and c_node < node:
            c_node = c_node.next

        if c_node == self.tail:
            self.tail = node

        if c_node is None:
            node.prev = self.head
            self.head.next = node
            self.head = node
        else:
            node.prev = c_node.prev
            node.next = c_node
            c_node.prev = node
        self.n_elem += 1

    def delete(self, value):
        if isinstance(value, Node):
            if value == self.tail:
                self.tail = self.tail.next
            elif value == self.head:
                self.head = self.head.prev
            del value
            self.n_elem -= 1
        elif isinstance(value, int):
            self.delete_by_id(value)
        else:
            return

    def delete_by_id(self, id):
        c_node = self.tail
        while c_node != None and c_node.id != id:
            c_node = c_node.next

        if c_node == self.tail:
            self.tail = self.tail.next
        elif c_node == self.head:
            self.head = self.head.prev
        if c_node is not None:
            del c_node
        self.n_elem -= 1