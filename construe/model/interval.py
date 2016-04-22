# -*- coding: utf-8 -*-
"""Module defining Interval model and operations"""
#ActiveState recipe 576816

class Interval(object):
    """
    Represents an interval.
    Defined as closed interval [start,end), which includes the start and
    end positions.
    Start and end do not have to be numeric types.
    """

    __slots__ = ('_start', '_end')

    def __init__(self, start, end):
        "Construct, start must be <= end."
        if start > end:
            raise ValueError('Start (%s) must not be greater than end (%s)'
                              % (start, end))
        self._start = start
        self._end = end

    @property
    def start(self):
        """The interval's start"""
        return self._start

    @property
    def end(self):
        """The interval's end"""
        return self._end


    def __str__(self):
        "As string."
        return '[%s,%s]' % (self.start, self.end)


    def __repr__(self):
        "String representation."
        return '[%s,%s]' % (self.start, self.end)


    def __cmp__(self, other):
        "Compare."
        if None == other:
            return 1
        start_cmp = cmp(self.start, other.start)
        if 0 != start_cmp:
            return start_cmp
        else:
            return cmp(self.end, other.end)


    def __hash__(self):
        "Hash."
        return hash(self.start) ^ hash(self.end)


    def intersection(self, other):
        "Intersection. @return: An empty intersection if there is none."
        if self > other:
            other, self = self, other
        if self.end <= other.start:
            return Interval(self.start, self.start)
        return Interval(other.start, min(self.end, other.end))

    def hull(self, other):
        "@return: Interval containing both self and other."
        if self > other:
            other, self = self, other
        return Interval(self.start, max(self.end, other.end))

    def overlap(self, other):
        "@return: True iff self intersects other."
        if self > other:
            other, self = self, other
        return self.end > other.start

    def overlapm(self, other):
        "@return: True iff selfs overlaps or meets other."
        if self > other:
            other, self = self, other
        return self.end >= other.start

    def move(self, offset):
        "@return: Interval displaced offset to start and end"
        return Interval(self.start+offset, self.end+offset)

    def __contains__(self, item):
        "@return: True iff item in self."
        return self.start <= item and item <= self.end

    @property
    def zero_in(self):
        "@return: True iff 0 in self."
        return self.start <= 0 and 0 <= self.end


    def subset(self, other):
        "@return: True iff self is subset of other."
        return self.start >= other.start and self.end <= other.end


    def proper_subset(self, other):
        "@return: True iff self is proper subset of other."
        return self.start > other.start and self.end < other.end

    @property
    def empty(self):
        "@return: True iff self is empty."
        return self.start == self.end

    @property
    def length(self):
        """@return: Difference between end and start"""
        return self.end - self.start

    @property
    def singleton(self):
        "@return: True iff self.end - self.start == 1."
        return self.end - self.start == 1


    def separation(self, other):
        "@return: The distance between self and other."
        if self > other:
            other, self = self, other
        if self.end > other.start:
            return 0
        else:
            return other.start - self.end


