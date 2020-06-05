# List and Set Experiments
An attempt to recreate the dynamism performance of C-like sets and lists (see Sedgewick book on [Algorithms in Java](https://algs4.cs.princeton.edu/home/) in [C++](https://www.amazon.com/Algorithms-Parts-1-4-Fundamentals-Structure/dp/0201350882/)) in Python, to circumvent the performance malus of NumPy-like dense matrices and (fixed) arrays.

I try to use as much established libraries in Python (see 'bisect', 'sortedcollections' and 'sortedcontainers'), while preserving links to C-interaction (preferably just-in-time via CTypes).

first results:

```
Real list created.
Time adding 131072 particles (RealList): 13.388341611000001
Time delete and insert 131072 particles (RealList): 4.802947877000001
Deleting 131072 elements ...
===========================================================================
Ordered list created.
Time adding 131072 particles (OrderedList): 13.890868412
Time delete and insert 131072 particles (OrderedList): 5.314203552999999
Deleting 131072 elements ...
===========================================================================
Time adding 131072 particles (NumPy Array): 12.663392634999994
Time delete and insert 131072 particles (NumPy Array): 52.743404987999995
```

for beyond 1,000,000 entities, numbers are as follows:
```
Real list created.
Time adding 1048576 particles (RealList): 123.48088695000001
Time delete and insert 1048576 particles (RealList): 47.50684311899994
Deleting 1048576 elements ...
===========================================================================
Ordered list created.
Time adding 1048576 particles (OrderedList): 119.61499352699991
Time delete and insert 1048576 particles (OrderedList): 51.41910386099994
Deleting 1048576 elements ...
===========================================================================
Time adding 1048576 particles (NumPy Array): 1282.5754407159998
Time delete and insert 1048576 particles (NumPy Array): 5043.001804089
```

when actually filling the nodes with particle data (e.g. Parcels' particles), the performance for 2e16 particles is as follows:
```
Real list created.
Time adding 262144 particles (RealList): 77.703234879
Time delete and insert 262144 particles (RealList): 10.690429034000005
Deleting 262144 elements ...
===========================================================================
Ordered list created.
Time adding 262144 particles (OrderedList): 77.52311503000001
Time delete and insert 262144 particles (OrderedList): 11.743454558999986
Deleting 262144 elements ...
===========================================================================
NumPy nD-Array created.
Time adding 262144 particles (NumPy Array): 559.621542179
Time delete and insert 262144 particles (NumPy Array): 3913.6630525410005
```

As we can see, add-and-remove actions arbitrarily in the list are done very fast (as natural to linked lists and sets),
while the fixed arrays and dense matrices in NumPy need to re-create and re-allocate memory all the time, hence the performance malus.
