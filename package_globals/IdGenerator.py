#import random
from numpy import random

class IdGenerator:
    released_ids = []
    next_id = 0

    def __init__(self):
        self.released_ids = []
        self.next_id = 0

    def nextID(self):
        n = len(self.released_ids)
        if n == 0:
            result = self.next_id
            self.next_id += 1
            return int(result)
        else:
            result = self.released_ids.pop(n-1)
            return int(result)

    def releaseID(self, id):
        self.released_ids.append(id)

    def preGenerateIDs(self, high_value):
        if len(self.released_ids) > 0:
            self.released_ids.clear()
        #for i in range(0, high_value):
        #    self.released_ids.append(i)
        self.released_ids = [i for i in range(0, high_value)]
        self.next_id = high_value

    def permuteIDs(self):
        n = len(self.released_ids)
        indices = random.randint(0, n, 2*n)
        for index in indices:
            id = self.released_ids.pop(index)
            self.released_ids.append(id)

        #for iter in range(0, 2*n):
        #    index = random.randint(0, n)
        #    id = self.released_ids.pop(index)
        #    self.released_ids.append(id)

    def __len__(self):
        return self.next_id