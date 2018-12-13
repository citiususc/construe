# -*- coding: utf-8 -*-
"""

This module includes two implementations of the Rame-Douglas-Peucker algorithm,
one general for sequences of arbitrary 2-D points, and another for a specific
input of consecutive point values in a numpy array structure.

Examples:
>>> line = [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1),(0,0)]
>>> RDP(line, 1.0)
[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]

>>> line = [(0,0),(0.5,0.5),(1,0),(1.25,-0.25),(1.5,.5)]
>>> RDP(line, 0.25)
[(0, 0), (0.5, 0.5), (1.25, -0.25), (1.5, 0.5)]

"""

import math
import numpy as np
import sortedcontainers


def array2points(array):
    """
    Creates a list of pairs (x,y) from an array of y values, to make it
    suitable to be processed by the RDP function.
    """
    return [(i, array[i]) for i in range(len(array))]


def arrayRDP(arr, epsilon=0.0, n=None):
    """
    This is a slightly modified version of the _aRDP function, that accepts
    as arguments the tolerance in the distance and the maximum number of points
    the algorithm can select.
    **Note:** The results of this algoritm should be identical to the arrayRDP
    function if the *n* parameter is not specified. In that case, the
    performance is slightly worse, although the asymptotic complexity is the
    same. For this reason, this function internally delegates the solution in
    that function if the *n* parameter is missing.

    Parameters
    ----------
    arr:
        Array of values of consecutive points.
    epsilon:
        Maximum difference allowed in the simplification process.
    n:
        Maximum number of points of the resulted simplificated array.

    Returns
    -------
    out:
        Array of indices of the selected points.
    """
    if n is None:
        return _aRDP(arr, epsilon)
    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    n = n or len(arr)
    if n < 3:
        return arr
    fragments = sortedcontainers.SortedDict()
    #We store the distances as negative values due to the default order of
    #sorteddict
    dist, idx = max_vdist(arr, 0, len(arr) - 1)
    fragments[(-dist, idx)] = (0, len(arr) - 1)
    while len(fragments) < n-1:
        (dist, idx), (first, last) = fragments.popitem(index=0)
        if -dist <= epsilon:
            #We have to put again the last item to prevent loss
            fragments[(dist, idx)] = (first, last)
            break
        else:
            #We have to break the fragment in the selected index
            dist, newidx = max_vdist(arr, first, idx)
            fragments[(-dist, newidx)] = (first, idx)
            dist, newidx = max_vdist(arr, idx, last)
            fragments[(-dist, newidx)] = (idx, last)
    #Now we have to get all the indices in the keys of the fragments in order.
    result = sortedcontainers.SortedList(i[0] for i in fragments.values())
    result.add(len(arr) - 1)
    return np.array(result)

def max_vdist(arr, first, last):
    """
    Obtains the distance and the index of the point in *arr* with maximum
    vertical distance to the line delimited by the first and last indices. The
    returned value is a tuple (dist, index).
    """
    if first == last:
        return (0.0, first)
    frg = arr[first:last+1]
    leng = last-first+1
    dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1],
                                                            [frg[0], frg[-1]]))
    idx = np.argmax(dist)
    return (dist[idx], first+idx)


def _aRDP(arr, epsilon):
    """
    Performs an optimized version of the RDP algorithm assuming as an input
    an array of single values, considered consecutive points, and **taking
    into account only the vertical distances**.
    """
    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    if len(arr) < 3:
        return arr
    result = set()
    stack = [(0, len(arr) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, idx = max_vdist(arr, first, last)
        if max_dist > epsilon:
            stack.extend([(first, idx),(idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))


def RDP(pts, epsilon):
    """
    This implementation of the Ramer-Douglas-Peucker algorithm has been taken
    from the javascript version in http://karthaus.nl/rdp/js/rdp.js.
    """
    if len(pts) < 3:
        return pts
    first = pts[0]
    last = pts[-1]
    idx = -1
    max_dist = 0
    for i in range(1, len(pts)-1):
        #We obtain the perpendicular distance of each point to the line
        #delimited by the two points.
        if first[0] == last[0]:
            dist = abs(pts[i][0] - first[0])
        else:
            slope = (last[1] - first[1]) / float(last[0] -first[0])
            intercept = first[1] - (slope * first[0])
            dist = (abs(slope * pts[i][0] - pts[i][1] + intercept)
                                                / math.sqrt(slope * slope + 1))
        if dist > max_dist:
            max_dist = dist
            idx = i
    if max_dist > epsilon:
        left = RDP(pts[:idx + 1], epsilon)
        right = RDP(pts[idx:], epsilon)
        return left[:-1] + right
    else:
        return [first, last]


if __name__ == "__main__":
    pass
