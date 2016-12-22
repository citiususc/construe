# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Wed Oct 29 16:06:46 2014

This module defines a class to allow the inclusion of the *hasnext()*
functionality to python iterators. The implementation was adopted from

http://stackoverflow.com/questions/1966591/hasnext-in-python-iterators

@author: T. Teijeiro
"""

class PredictableIter(object):
    """
    This class supports the iteration protocol, adding the common routine (in
    other languages) *hasnext()* to allow the pre-checking of iteration
    capabilities. This implementations minimizes the number of pre-generated
    items, forwarding the *next()* call when possible.
    """

    #We may create a lot of iterators, so we have to keep low mem consumption.
    __slots__ = ('iter', '_hasnext', '_thenext')

    def __init__(self, iterator):
        """Initializes a new wrapper with an existing iterator"""
        self.iter = iter(iterator)
        self._hasnext = self._thenext = None

    def __iter__(self):
        """Returns the iterator object"""
        return self

    def next(self):
        """Obtains the next element in the sequence"""
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.iter)
        self._hasnext = None
        return result

    def hasnext(self):
        """Checks if the iterator has more elements to return."""
        if self._hasnext is None:
            try:
                self._thenext = next(self.iter)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext
