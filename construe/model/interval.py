# -*- coding: utf-8 -*-
"""Module defining Interval model and operations"""
#Based on ActiveState recipe 576816

class Interval(object):
    """
    Represents an interval.
    Defined as closed interval [start,end), which includes the start and
    end positions.
    Start and end do not have to be numeric types.
    """

    __slots__ = ('_start', '_end')

    def __init__(self, start, end):
        "Construct, start must be <= end"
        self.set(start, end)

    def set(self, start, end):
        """Sets the limits of this interval, start must be <= end"""
        if start > end:
            raise ValueError('Start ({0}) must not be greater than end '
                             '({1})'.format(start, end))
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
        return '[{0},{1}]'.format(self._start, self._end)


    def __repr__(self):
        "String representation."
        return '[{0},{1}]'.format(self._start, self._end)


    def __cmp__(self, other):
        "Compare."
        if other is None:
            return 1
        start_cmp = cmp(self._start, other.start)
        return start_cmp if start_cmp != 0 else cmp(self._end, other.end)


    def cpy(self, other):
        """Sets the limits of this interval, by copying them from another one"""
        self.set(other.start, other.end)

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
        return Interval(self._start+offset, self._end+offset)

    def __contains__(self, item):
        "@return: True iff item in self."
        return self._start <= item <= self._end

    def subset(self, other):
        "@return: True iff self is subset of other."
        return self._start >= other.start and self._end <= other.end


    def proper_subset(self, other):
        "@return: True iff self is proper subset of other."
        return self._start > other.start and self._end < other.end

    @property
    def zero_in(self):
        "@return: True iff 0 in self."
        return self._start <= 0 <= self._end

    @property
    def empty(self):
        "@return: True iff self is empty."
        return self._start == self._end

    @property
    def length(self):
        """@return: Difference between end and start"""
        return self._end - self._start

    @property
    def singleton(self):
        "@return: True iff self.end - self.start == 1."
        return self._end - self._start == 1


    def separation(self, other):
        "@return: The distance between self and other."
        if self > other:
            other, self = self, other
        return 0 if self.end > other.start else other.start - self.end
